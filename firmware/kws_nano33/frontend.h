/* frontend.h — MFCC + delta + delta-delta + CMVN of the KWS model, in plain C.
 *
 * The same code runs on the board and, compiled by tools/build_frontend_pc.py, on the PC, so that the accuracy of the device
 * arithmetic can be measured on the whole test set. Specification: tools/frontend_ref.py and knowledge base 03.
 *
 * Two parts, so that audio can be processed while it arrives:
 *   fe_push_frame()     per 10 ms frame: window, power spectrum, mel filterbank, log; one row of the log-mel ring.
 *   fe_finish_window()  per decision: top-dB clip, DCT, deltas, zero row, CMVN, int8 quantization of the last 97 rows.
 */
#ifndef KWS_FRONTEND_H_
#define KWS_FRONTEND_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  FE_FRAME_SAMPLES = 512,   /* samples read per frame; consecutive frames start 160 samples apart */
  FE_OUT_BYTES = 98 * 39,   /* int8 model input, row by row */
  FE_CLIP_BYTES = 32000,    /* one second of int16 audio */
  FE_LOGMEL_ROWS = 97,
  FE_LOGMEL_COLS = 40
};

typedef struct {
  float logmel[FE_LOGMEL_ROWS][FE_LOGMEL_COLS];  /* ring of log-mel rows */
  uint32_t frames;                               /* frames pushed since fe_reset() */
} fe_state_t;

void fe_reset(fe_state_t* s);

/* One frame of FE_FRAME_SAMPLES int16 samples. */
void fe_push_frame(fe_state_t* s, const int16_t* samples);

/* Number of complete windows available: 1 once 97 frames have been pushed, otherwise 0. */
int fe_window_ready(const fe_state_t* s);

/* Features of the last 97 frames into out[FE_OUT_BYTES]. feat_debug may be NULL; otherwise it receives the 98x39 floats before
 * quantization. Uses about 5.5 KB of stack and does not change the state. Call only when fe_window_ready(). */
void fe_finish_window(const fe_state_t* s, int8_t* out, float* feat_debug);

/* Convenience for a whole clip of 16000 samples: fe_reset(), 97 frames, fe_finish_window(). */
void fe_process_clip(fe_state_t* s, const int16_t* pcm, int8_t* out, float* feat_debug);

/* Power spectrum of one windowed frame, 257 bins. Exposed for tests. */
void fe_power_spectrum(const float* frame, float* power);

/* Copies the 97 log-mel rows oldest first into out[97 * 40]. For tests and for the LOGMEL serial command. */
void fe_get_logmel(const fe_state_t* s, float* out);

/* sizeof(fe_state_t), for the ctypes wrapper. */
uint32_t fe_state_size(void);

#ifdef __cplusplus
}
#endif

#endif /* KWS_FRONTEND_H_ */
