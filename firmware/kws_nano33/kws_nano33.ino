// kws_nano33.ino — KWS firmware. So far it only runs board tests (no microphone yet); KWS_TEST_MODE in config.h selects one:
//   1  selftest_model.cpp     model only: V1 on golden vectors, Invoke benchmark, per-operator profile (M2)
//   2  selftest_frontend.cpp  frontend.c + model on built-in golden audio: features for the PC comparison, timing (M3)
//   3  serial_proto.cpp       serial test server: the PC sends audio or features, the board answers (M3, tools/serial_harness.py)
// After the test the report is repeated every 30 s, so a host that opens the port late still gets it.
// Board: arduino:mbed_nano:nano33ble, core 4.6.0. Library: LiteRT_Micro, generated from tensorflow/tflite-micro (see README.md).

#include <Arduino.h>
#include <litert_micro.h>     // must come first: this include is what makes the builder find the library

#include "kws_model.h"
#include "selftest.h"
#include "tensorflow/lite/micro/system_setup.h"

static const uint32_t kHostWaitMs = 15000;
static const uint32_t kReportMs = 30000;

void selftestStep(const char* what) {
  Serial.print("STEP "); Serial.println(what);
  Serial.flush();
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  const uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < kHostWaitMs) {}
  selftestStep("target_init");
  tflite::micro::InitializeTarget();

  selftestStep("begin");
  const uint32_t t = micros();
  const kws_model::Status status = kws_model::begin();
  selftestRun(status, micros() - t);
}

void loop() {
  selftestReport();
  delay(kReportMs);
}
