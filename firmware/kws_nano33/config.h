// config.h — build-time settings of the KWS firmware. Filled milestone by milestone (see the project plan).
#pragma once

// Tensor arena: arena_used_bytes() of the current model and library (133,688 B, M2) + 2 KB. begin() fails with
// kAllocateFailed if a new model needs more. The rule from M1 step 4 stays: static RAM ≤ 252,416 B, heap region ≥ 8 KB.
#ifndef KWS_ARENA_BYTES
#define KWS_ARENA_BYTES (133688 + 2048)
#endif

// 1 = time every operator with the DWT cycle counter. Costs a virtual call per operator, 14 per Invoke.
#ifndef KWS_PROFILE
#define KWS_PROFILE 1
#endif

// Which board test the firmware runs: 1 = model (selftest_model.cpp), 2 = frontend (selftest_frontend.cpp),
// 3 = serial test server (serial_proto.cpp, driven by tools/serial_harness.py).
#ifndef KWS_TEST_MODE
#define KWS_TEST_MODE 3
#endif
