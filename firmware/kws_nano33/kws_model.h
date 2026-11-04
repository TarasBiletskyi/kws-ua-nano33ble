// kws_model.h — wrapper around LiteRT for Microcontrollers. All calls into the runtime live in kws_model.cpp.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace kws_model {

constexpr int kInputBytes = 98 * 39;  // int8 features, row by row
constexpr int kNumClasses = 15;
constexpr int kGapSize = 64;
constexpr int kMaxOps = 16;           // the model has 14 operators

enum class Status {
  kOk = 0,
  kSchemaMismatch,      // model schema differs from the runtime's
  kInterpreterInit,     // MicroInterpreter constructor failed (unknown operator in the model, for example)
  kAllocateFailed,      // AllocateTensors failed (arena too small, for example)
  kUnexpectedTensors,   // input or outputs have other shapes or types than this firmware expects
  kInvokeFailed,
};

struct QuantParams {
  float scale;
  int zero_point;
};

// Builds the interpreter and allocates the tensors. Runtime errors also go to Serial through MicroPrintf.
Status begin();
Status invoke();

int8_t* input();                 // kInputBytes
const int8_t* probabilities();   // kNumClasses, softmax
const int8_t* gapVector();       // kGapSize, global average pooling

QuantParams inputQuant();
QuantParams probabilitiesQuant();
QuantParams gapQuant();

uint32_t schemaVersion();        // of the model
size_t arenaCapacity();
size_t arenaUsed();              // valid after begin()

// DWT cycle counter (started by begin()) and the cycles of the last invoke().
uint32_t cycleCounter();
uint32_t lastInvokeCycles();

// Per-operator cycles of the last invoke(), in graph order. opCount() is 0 when KWS_PROFILE is 0.
int opCount();
const char* opTag(int i);
uint32_t opCycles(int i);

}  // namespace kws_model
