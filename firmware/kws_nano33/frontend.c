/* frontend.c — see frontend.h. Plain C99, no Arduino dependency. All arithmetic is float32, except the running sums of CMVN. */
#include "frontend.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define FE_TABLES_NO_CLASS_NAMES
#include "fe_tables.h"

#define FE_FFT_SIZE (FE_N_FFT / 2) /* the real 512-point FFT is computed through a complex 256-point FFT */

/* Radix-2 decimation-in-time FFT of 256 complex values, followed by the real-FFT split: power of bins 0..256. */
void fe_power_spectrum(const float* frame, float* power) {
  float zr[FE_FFT_SIZE];
  float zi[FE_FFT_SIZE];
  int i, k, m;

  for (i = 0; i < FE_FFT_SIZE; ++i) {
    const int j = FE_FFT_BITREV[i];
    zr[j] = frame[2 * i];
    zi[j] = frame[2 * i + 1];
  }

  for (m = 2; m <= FE_FFT_SIZE; m *= 2) {
    const int half = m / 2;
    const int tw_step = 2 * (FE_FFT_SIZE / m); /* the table holds exp(-2*pi*i*k/512) */
    for (k = 0; k < half; ++k) {
      const float wr = FE_FFT_TW[2 * (k * tw_step)];
      const float wi = FE_FFT_TW[2 * (k * tw_step) + 1];
      for (i = k; i < FE_FFT_SIZE; i += m) {
        const int j = i + half;
        const float tr = wr * zr[j] - wi * zi[j];
        const float ti = wr * zi[j] + wi * zr[j];
        zr[j] = zr[i] - tr;
        zi[j] = zi[i] - ti;
        zr[i] += tr;
        zi[i] += ti;
      }
    }
  }

  {
    const float dc = zr[0] + zi[0];
    const float nyq = zr[0] - zi[0];
    power[0] = dc * dc;
    power[FE_FFT_SIZE] = nyq * nyq;
  }
  for (k = 1; k <= FE_FFT_SIZE / 2; ++k) {
    const int n = FE_FFT_SIZE - k;
    const float er = 0.5f * (zr[k] + zr[n]);
    const float ei = 0.5f * (zi[k] - zi[n]);
    /* odd part: -i * (Z[k] - conj(Z[n])) / 2 */
    const float or_ = 0.5f * (zi[k] + zi[n]);
    const float oi = -0.5f * (zr[k] - zr[n]);
    const float wr = FE_FFT_TW[2 * k];
    const float wi = FE_FFT_TW[2 * k + 1];
    const float tr = wr * or_ - wi * oi;
    const float ti = wr * oi + wi * or_;
    const float ar = er + tr;
    const float ai = ei + ti;
    const float br = er - tr;
    const float bi = ei - ti;
    power[k] = ar * ar + ai * ai;
    power[n] = br * br + bi * bi;
  }
}

void fe_reset(fe_state_t* s) {
  memset(s, 0, sizeof(*s));
}

void fe_push_frame(fe_state_t* s, const int16_t* samples) {
  float frame[FE_N_FFT];
  float power[FE_N_BINS];
  float* row = s->logmel[s->frames % FE_LOGMEL_ROWS];
  int i, m;

  memset(frame, 0, sizeof(frame));
  for (i = 0; i < FE_WIN_LEN; ++i) {
    frame[FE_WIN_OFFSET + i] = ((float)samples[FE_WIN_OFFSET + i] / 32768.0f) * FE_WINDOW[i];
  }
  fe_power_spectrum(frame, power);

  for (m = 0; m < FE_N_MELS; ++m) {
    const int start = MEL_START[m];
    const int len = MEL_LEN[m];
    const float* w = &MEL_W[MEL_OFFSET[m]];
    float sum = 0.0f;
    for (i = 0; i < len; ++i) sum += power[start + i] * w[i];
    row[m] = 10.0f * log10f(sum > FE_AMIN ? sum : FE_AMIN);
  }
  s->frames++;
}

int fe_window_ready(const fe_state_t* s) {
  return s->frames >= FE_LOGMEL_ROWS;
}

static const float* logmel_row(const fe_state_t* s, int t) {
  /* t = 0 is the oldest of the last 97 frames */
  return s->logmel[(s->frames + (uint32_t)t) % FE_LOGMEL_ROWS];
}

void fe_get_logmel(const fe_state_t* s, float* out) {
  int t;
  for (t = 0; t < FE_LOGMEL_ROWS; ++t) {
    memcpy(out + t * FE_LOGMEL_COLS, logmel_row(s, t), FE_LOGMEL_COLS * sizeof(float));
  }
}

/* Savitzky-Golay derivative of one MFCC column with interp edges, as in tools/frontend_ref.py::sg_filter. */
static void sg_column(const float* mfcc, int coeff, const float* center, const float* head, const float* tail, float* out) {
  const int n = FE_N_FRAMES_RAW;
  int t, j;
  for (t = 0; t < n; ++t) {
    const float* kernel;
    int first;
    float sum = 0.0f;
    if (t < FE_SG_HALF) {
      kernel = head + t * FE_SG_WIDTH;
      first = 0;
    } else if (t >= n - FE_SG_HALF) {
      kernel = tail + (t - (n - FE_SG_HALF)) * FE_SG_WIDTH;
      first = n - FE_SG_WIDTH;
    } else {
      kernel = center;
      first = t - FE_SG_HALF;
    }
    for (j = 0; j < FE_SG_WIDTH; ++j) sum += kernel[j] * mfcc[(first + j) * FE_N_MFCC + coeff];
    out[t] = sum;
  }
}

/* CMVN over 97 values plus the zero row, then int8. Population standard deviation + eps, round to nearest even, clip. */
static void normalize_column(const float* v, int col, int8_t* out, float* feat_debug) {
  double sum = 0.0;
  double acc = 0.0;
  float mean, std;
  int t;
  for (t = 0; t < FE_N_FRAMES_RAW; ++t) sum += (double)v[t];
  mean = (float)(sum / FE_N_FRAMES);
  for (t = 0; t < FE_N_FRAMES_RAW; ++t) {
    const float d = v[t] - mean;
    acc += (double)d * (double)d;
  }
  acc += (double)mean * (double)mean; /* the zero row */
  std = (float)sqrt(acc / FE_N_FRAMES) + FE_CMVN_EPS;

  for (t = 0; t < FE_N_FRAMES; ++t) {
    const float x = t < FE_N_FRAMES_RAW ? v[t] : 0.0f;
    const float f = (x - mean) / std;
    const float q = rintf(f / FE_IN_SCALE) + (float)FE_IN_ZP;
    int8_t r;
    if (q >= 127.0f) r = 127;
    else if (q <= -128.0f) r = -128;
    else r = (int8_t)q;
    out[t * FE_N_FEAT + col] = r;
    if (feat_debug != NULL) feat_debug[t * FE_N_FEAT + col] = f;
  }
}

void fe_finish_window(const fe_state_t* s, int8_t* out, float* feat_debug) {
  float mfcc[FE_N_FRAMES_RAW * FE_N_MFCC];
  float column[FE_N_FRAMES_RAW];
  float top = -1e30f;
  float floor_db;
  int t, n, i;

  for (t = 0; t < FE_LOGMEL_ROWS; ++t) {
    const float* row = s->logmel[t];
    for (i = 0; i < FE_N_MELS; ++i) {
      if (row[i] > top) top = row[i];
    }
  }
  floor_db = top - FE_TOP_DB;

  for (t = 0; t < FE_N_FRAMES_RAW; ++t) {
    const float* row = logmel_row(s, t);
    for (n = 0; n < FE_N_MFCC; ++n) {
      const float* d = &DCT13x40[n * FE_N_MELS];
      float sum = 0.0f;
      for (i = 0; i < FE_N_MELS; ++i) sum += d[i] * (row[i] > floor_db ? row[i] : floor_db);
      mfcc[t * FE_N_MFCC + n] = sum;
    }
  }

  for (n = 0; n < FE_N_MFCC; ++n) {
    for (t = 0; t < FE_N_FRAMES_RAW; ++t) column[t] = mfcc[t * FE_N_MFCC + n];
    normalize_column(column, n, out, feat_debug);
    sg_column(mfcc, n, SG1_CENTER, SG1_HEAD, SG1_TAIL, column);
    normalize_column(column, FE_N_MFCC + n, out, feat_debug);
    sg_column(mfcc, n, SG2_CENTER, SG2_HEAD, SG2_TAIL, column);
    normalize_column(column, 2 * FE_N_MFCC + n, out, feat_debug);
  }
}

void fe_process_clip(fe_state_t* s, const int16_t* pcm, int8_t* out, float* feat_debug) {
  int t;
  fe_reset(s);
  for (t = 0; t < FE_N_FRAMES_RAW; ++t) fe_push_frame(s, pcm + t * FE_HOP);
  fe_finish_window(s, out, feat_debug);
}

uint32_t fe_state_size(void) {
  return (uint32_t)sizeof(fe_state_t);
}
