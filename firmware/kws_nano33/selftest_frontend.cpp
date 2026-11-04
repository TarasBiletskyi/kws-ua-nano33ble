// selftest_frontend.cpp — M3 test of frontend.c on the board (KWS_TEST_MODE 2), test V2 on the built-in golden audio.
// For each of the 3 clips: 97 frames + window finish, timed in CPU cycles; then the features as exact float32 hex for the comparison
// with the PC (tools/m3_board_compare.py); then one Invoke on those features.
// Report lines: STEP, FEATF (once), HELLO, ALLOC, FE, INVOKE, PROB8, MEM.
#include "config.h"
#if KWS_TEST_MODE == 2

#include <Arduino.h>
#include <mbed.h>
#include "platform/mbed_stats.h"
#include <string.h>

#include "frontend.h"
#include "golden_set.h"
#include "kws_model.h"
#include "model_data.h"
#include "selftest.h"

struct ClipResult {
  uint32_t frame_min, frame_max, finish;
  uint64_t frame_sum;
  uint32_t invoke_cycles;
  int status, top;
  int8_t probs[kws_model::kNumClasses];
};

static fe_state_t g_fe;
static int8_t g_input[FE_OUT_BYTES];
static float g_feat[FE_OUT_BYTES];
static ClipResult g_clip[GOLDEN_N_AUDIO];
static kws_model::Status g_begin_status = kws_model::Status::kOk;
static uint32_t g_begin_us = 0;
static uint32_t g_report_n = 0;
static bool g_ran = false;

static void printFeatures(int clip) {
  for (int r = 0; r < 98; ++r) {
    Serial.print("FEATF clip="); Serial.print(clip);
    Serial.print(" row="); Serial.print(r);
    for (int c = 0; c < 39; ++c) {
      uint32_t bits;
      memcpy(&bits, &g_feat[r * 39 + c], sizeof(bits));
      Serial.print(c ? ',' : ' ');
      Serial.print(bits, HEX);
    }
    Serial.println();
  }
}

static void runClip(int clip) {
  ClipResult& r = g_clip[clip];
  const int16_t* pcm = GOLDEN_AUDIO[clip];
  r.frame_min = 0xFFFFFFFFu; r.frame_max = 0; r.frame_sum = 0;
  fe_reset(&g_fe);
  for (int t = 0; t < FE_LOGMEL_ROWS; ++t) {
    const uint32_t c0 = kws_model::cycleCounter();
    fe_push_frame(&g_fe, pcm + t * 160);
    const uint32_t c = kws_model::cycleCounter() - c0;
    if (c < r.frame_min) r.frame_min = c;
    if (c > r.frame_max) r.frame_max = c;
    r.frame_sum += c;
  }
  const uint32_t c0 = kws_model::cycleCounter();
  fe_finish_window(&g_fe, g_input, nullptr);
  r.finish = kws_model::cycleCounter() - c0;

  fe_finish_window(&g_fe, g_input, g_feat);  // the same again, this time keeping the floats for the PC comparison
  printFeatures(clip);

  memcpy(kws_model::input(), g_input, FE_OUT_BYTES);
  r.status = (int)kws_model::invoke();
  r.invoke_cycles = kws_model::lastInvokeCycles();
  memcpy(r.probs, kws_model::probabilities(), kws_model::kNumClasses);
  int best = 0;
  for (int i = 1; i < kws_model::kNumClasses; ++i) if (r.probs[i] > r.probs[best]) best = i;
  r.top = best;
}

void selftestRun(kws_model::Status begin_status, uint32_t begin_us) {
  g_begin_status = begin_status;
  g_begin_us = begin_us;
  if (begin_status != kws_model::Status::kOk) return;
  selftestStep("clips");
  for (int clip = 0; clip < GOLDEN_N_AUDIO; ++clip) {
    runClip(clip);
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }
  g_ran = true;
  selftestStep("done");
}

void selftestReport() {
  ++g_report_n;
  Serial.print("HELLO kws-ua-nano33ble kws_nano33 m3-frontend v1 lib=85c6999 report="); Serial.print(g_report_n);
  Serial.print(" model="); Serial.print(KWS_MODEL_SHA256_8);
#if defined(__OPTIMIZE_SIZE__)
  Serial.println(" opt=size");
#else
  Serial.println(" opt=speed");
#endif
  Serial.print("ALLOC status="); Serial.print((int)g_begin_status);
  Serial.print(" us="); Serial.print(g_begin_us);
  Serial.print(" arena_capacity="); Serial.print((uint32_t)kws_model::arenaCapacity());
  Serial.print(" arena_used="); Serial.println((uint32_t)kws_model::arenaUsed());
  if (!g_ran) {
    Serial.println("ERR test not run, see the ALLOC status and the runtime messages above");
    return;
  }
  for (int clip = 0; clip < GOLDEN_N_AUDIO; ++clip) {
    const ClipResult& r = g_clip[clip];
    Serial.print("FE clip="); Serial.print(clip);
    Serial.print(" frame_cycles_min="); Serial.print(r.frame_min);
    Serial.print(" frame_cycles_mean="); Serial.print((uint32_t)(r.frame_sum / FE_LOGMEL_ROWS));
    Serial.print(" frame_cycles_max="); Serial.print(r.frame_max);
    Serial.print(" finish_cycles="); Serial.println(r.finish);
    Serial.print("INVOKE clip="); Serial.print(clip);
    Serial.print(" status="); Serial.print(r.status);
    Serial.print(" cycles="); Serial.print(r.invoke_cycles);
    Serial.print(" top="); Serial.print(r.top);
    Serial.print(" label="); Serial.println((int)GOLDEN_LABELS[clip]);
    Serial.print("PROB8 clip="); Serial.print(clip);
    for (int i = 0; i < kws_model::kNumClasses; ++i) { Serial.print(i ? ',' : ' '); Serial.print((int)r.probs[i]); }
    Serial.println();
  }
  mbed_stats_heap_t heap; mbed_stats_heap_get(&heap);
  Serial.print("MEM heap_current="); Serial.print(heap.current_size);
  Serial.print(" heap_reserved="); Serial.print(heap.reserved_size);
  mbed_stats_thread_t th[10];
  const size_t n = mbed_stats_thread_get_each(th, 10);
  for (size_t i = 0; i < n; ++i) {
    if (th[i].name && strcmp(th[i].name, "main") == 0) {
      Serial.print(" main_stack_used="); Serial.print(th[i].stack_size - th[i].stack_space);
    }
  }
  Serial.println();
}

#endif  // KWS_TEST_MODE == 2
