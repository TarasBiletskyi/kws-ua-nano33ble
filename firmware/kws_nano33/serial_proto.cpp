// serial_proto.cpp — serial test server (KWS_TEST_MODE 3): the PC sends audio or features, the board answers with its results.
// Driven by tools/serial_harness.py. Every request is a text line, optionally followed by binary data + CRC16 (CCITT, big endian):
//
//   PING                          PONG fw=... model=... in_scale= in_zp= gap_scale= gap_zp= arena_used= arena_capacity=
//   AUDIO 32000 [F]  + data       16000 int16 samples through the frontend and the model. Answer: FEAT8, optionally FEATF, PROB8, GAP8, TIME, TOP
//   FEATIN 3822      + data       3822 int8 features straight into the model. Answer: PROB8, GAP8, TIME, TOP
//   MEM                           heap and main-stack use
//
// Before binary data the board answers "READY" and the PC waits for it, so a failed request cannot leave data that looks like a command.
// Data travels in blocks of kBlockBytes, each acknowledged with "K": unbroken 32 KB transfers lost their last 62 bytes after a few requests
// (cause not identified; blocks up to 3.8 KB never failed).
// Binary answers: "<TAG> <n> <crc16 hex>\n" + n bytes. Text answers: one line. Each request ends with "END\n"; failures print "ERR <why>".
#include "config.h"
#if KWS_TEST_MODE == 3

#include <Arduino.h>
#include <mbed.h>
#include "platform/mbed_stats.h"
#include <string.h>

#include "frontend.h"
#include "kws_model.h"
#include "model_data.h"
#include "selftest.h"

static const int kAudioBytes = FE_CLIP_BYTES;
static const uint32_t kPayloadTimeoutMs = 5000;
static const size_t kBlockBytes = 2048;

static fe_state_t g_fe;
static int16_t g_audio[kAudioBytes / 2];
static float g_feat[FE_OUT_BYTES];
static char g_line[64];
static kws_model::Status g_begin_status = kws_model::Status::kOk;
static uint32_t g_begin_us = 0;

static uint16_t crc16(const uint8_t* data, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

static bool readLine() {
  size_t n = 0;
  for (;;) {
    while (!Serial.available()) { rtos::ThisThread::yield(); }
    const char c = (char)Serial.read();
    if (c == '\n') { g_line[n] = '\0'; return n > 0; }
    if (c != '\r' && n < sizeof(g_line) - 1) g_line[n++] = c;
  }
}

// Reads n bytes + CRC. On a failure prints ERR and returns false.
static bool receiveBlock(uint8_t* dst, size_t n) {
  // The driver moves arriving packets into the 256-byte ring of Serial from its event thread, so all data must be read through Serial.
  // That thread has the priority of this one: a busy wait starves it for a whole time slice per packet (≈8 KB/s), so yield when empty.
  size_t got = 0;
  uint32_t last = millis();
  while (got < n) {
    int avail = Serial.available();
    if (avail <= 0) {
      rtos::ThisThread::yield();
      if (millis() - last > kPayloadTimeoutMs) {
        Serial.print("ERR got "); Serial.print((uint32_t)got); Serial.print(" of "); Serial.println((uint32_t)n);
        return false;
      }
      continue;
    }
    while (avail-- > 0 && got < n) dst[got++] = (uint8_t)Serial.read();
    last = millis();
  }
  return true;
}

static bool readPayload(uint8_t* dst, size_t n) {
  Serial.println("READY");
  for (size_t off = 0; off < n; off += kBlockBytes) {
    const size_t block = (n - off) < kBlockBytes ? (n - off) : kBlockBytes;
    if (!receiveBlock(dst + off, block)) { Serial.println("ERR timeout"); return false; }
    Serial.println("K");
  }
  uint8_t c[2];
  if (!receiveBlock(c, 2)) { Serial.println("ERR timeout crc"); return false; }
  if (crc16(dst, n) != (uint16_t)((c[0] << 8) | c[1])) { Serial.println("ERR crc"); return false; }
  return true;
}

static void sendBinary(const char* tag, const uint8_t* data, size_t n) {
  Serial.print(tag); Serial.print(' '); Serial.print((uint32_t)n); Serial.print(' ');
  Serial.println(crc16(data, n), HEX);
  size_t off = 0;
  while (off < n) {
    const size_t chunk = (n - off) > 512 ? 512 : (n - off);
    Serial.write(data + off, chunk);
    off += chunk;
  }
}

static void printVector(const char* tag, const int8_t* v, int n) {
  Serial.print(tag);
  for (int i = 0; i < n; ++i) { Serial.print(i ? ',' : ' '); Serial.print((int)v[i]); }
  Serial.println();
}

static void runModelAndAnswer(uint32_t frames_cycles, uint32_t finish_cycles) {
  const kws_model::Status st = kws_model::invoke();
  if (st != kws_model::Status::kOk) { Serial.println("ERR invoke"); return; }
  const int8_t* p = kws_model::probabilities();
  printVector("PROB8", p, kws_model::kNumClasses);
  printVector("GAP8", kws_model::gapVector(), kws_model::kGapSize);
  Serial.print("TIME frames_cycles="); Serial.print(frames_cycles);
  Serial.print(" finish_cycles="); Serial.print(finish_cycles);
  Serial.print(" invoke_cycles="); Serial.println(kws_model::lastInvokeCycles());
  int best = 0;
  for (int i = 1; i < kws_model::kNumClasses; ++i) if (p[i] > p[best]) best = i;
  Serial.print("TOP "); Serial.println(best);
}

static void handleAudio(bool want_float) {
  if (!readPayload((uint8_t*)g_audio, kAudioBytes)) return;
  uint32_t c0 = kws_model::cycleCounter();
  fe_reset(&g_fe);
  for (int t = 0; t < FE_LOGMEL_ROWS; ++t) fe_push_frame(&g_fe, g_audio + t * 160);
  const uint32_t frames_cycles = kws_model::cycleCounter() - c0;
  c0 = kws_model::cycleCounter();
  fe_finish_window(&g_fe, kws_model::input(), nullptr);
  const uint32_t finish_cycles = kws_model::cycleCounter() - c0;

  sendBinary("FEAT8", (const uint8_t*)kws_model::input(), FE_OUT_BYTES);
  if (want_float) {
    int8_t* in = kws_model::input();
    fe_finish_window(&g_fe, in, g_feat);
    sendBinary("FEATF", (const uint8_t*)g_feat, sizeof(g_feat));
  }
  runModelAndAnswer(frames_cycles, finish_cycles);
}

static void handleFeatures() {
  if (!readPayload((uint8_t*)kws_model::input(), FE_OUT_BYTES)) return;
  runModelAndAnswer(0, 0);
}

static void handlePing() {
  const kws_model::QuantParams in = kws_model::inputQuant(), gap = kws_model::gapQuant();
  Serial.print("PONG fw=m3-serial-v1 lib=85c6999 model="); Serial.print(KWS_MODEL_SHA256_8);
  Serial.print(" in_scale="); Serial.print(in.scale, 8);
  Serial.print(" in_zp="); Serial.print(in.zero_point);
  Serial.print(" gap_scale="); Serial.print(gap.scale, 8);
  Serial.print(" gap_zp="); Serial.print(gap.zero_point);
  Serial.print(" arena_used="); Serial.print((uint32_t)kws_model::arenaUsed());
  Serial.print(" arena_capacity="); Serial.println((uint32_t)kws_model::arenaCapacity());
}

static void handleMem() {
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

void selftestRun(kws_model::Status begin_status, uint32_t begin_us) {
  g_begin_status = begin_status;
  g_begin_us = begin_us;
  if (begin_status != kws_model::Status::kOk) return;  // loop() keeps reporting the failure
  Serial.setTimeout(5000);
  selftestStep("serial server");
  for (;;) {
    if (!readLine()) continue;
    if (strcmp(g_line, "PING") == 0) handlePing();
    else if (strcmp(g_line, "MEM") == 0) handleMem();
    else if (strcmp(g_line, "AUDIO 32000") == 0) handleAudio(false);
    else if (strcmp(g_line, "AUDIO 32000 F") == 0) handleAudio(true);
    else if (strcmp(g_line, "FEATIN 3822") == 0) handleFeatures();
    else { Serial.print("ERR unknown "); Serial.println(g_line); }
    Serial.println("END");
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }
}

void selftestReport() {
  Serial.print("ERR begin status="); Serial.print((int)g_begin_status);
  Serial.print(" us="); Serial.println(g_begin_us);
}

#endif  // KWS_TEST_MODE == 3
