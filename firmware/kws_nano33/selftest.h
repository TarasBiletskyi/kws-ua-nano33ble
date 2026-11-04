// selftest.h — entry points of the board tests, one implementation per KWS_TEST_MODE (config.h).
#pragma once
#include <stdint.h>

#include "kws_model.h"

void selftestStep(const char* what);                                  // kws_nano33.ino: progress line, flushed
void selftestRun(kws_model::Status begin_status, uint32_t begin_us);  // once, after kws_model::begin()
void selftestReport();                                                // repeated from loop()
