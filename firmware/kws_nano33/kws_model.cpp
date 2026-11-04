// kws_model.cpp — LiteRT for Microcontrollers: the two-output DS-CNN (softmax + GAP vector) from model_data.cpp in flash.
#include "kws_model.h"

#include <litert_micro.h>

#include "config.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include <cmsis.h>  // DWT, CoreDebug

namespace kws_model {
namespace {

alignas(16) uint8_t g_arena[KWS_ARENA_BYTES];

const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_probs = nullptr;
TfLiteTensor* g_gap = nullptr;
uint32_t g_last_invoke_cycles = 0;

// The library's own MicroProfiler counts microseconds; this one counts CPU cycles through the same interface.
class CycleProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* tag) override {
    if (count_ >= kMaxOps) return kMaxOps;
    tags_[count_] = tag;
    cycles_[count_] = DWT->CYCCNT;
    return count_++;
  }
  void EndEvent(uint32_t handle) override {
    if (handle < (uint32_t)kMaxOps) cycles_[handle] = DWT->CYCCNT - cycles_[handle];
  }
  void reset() { count_ = 0; }
  int count() const { return (int)count_; }
  const char* tag(int i) const { return tags_[i]; }
  uint32_t cycles(int i) const { return cycles_[i]; }

 private:
  uint32_t count_ = 0;
  const char* tags_[kMaxOps] = {};
  uint32_t cycles_[kMaxOps] = {};
};

#if KWS_PROFILE
CycleProfiler g_profiler;
tflite::MicroProfilerInterface* const g_profiler_ptr = &g_profiler;
#else
tflite::MicroProfilerInterface* const g_profiler_ptr = nullptr;
#endif

void startCycleCounter() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

bool isInt8Vector(const TfLiteTensor* t, int size) {
  return t->type == kTfLiteInt8 && t->dims->size == 2 && t->dims->data[0] == 1 && t->dims->data[1] == size;
}

QuantParams quant(const TfLiteTensor* t) { return {t->params.scale, t->params.zero_point}; }

}  // namespace

Status begin() {
  startCycleCounter();
  g_model = tflite::GetModel(g_kws_model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) return Status::kSchemaMismatch;

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter interpreter(g_model, resolver, g_arena, sizeof(g_arena), nullptr, g_profiler_ptr);
  if (interpreter.initialization_status() != kTfLiteOk) return Status::kInterpreterInit;
  g_interpreter = &interpreter;

  if (interpreter.AllocateTensors() != kTfLiteOk) return Status::kAllocateFailed;

  g_input = interpreter.input(0);
  if (interpreter.inputs_size() != 1 || g_input->type != kTfLiteInt8 || g_input->bytes != (size_t)kInputBytes) {
    return Status::kUnexpectedTensors;
  }
  if (interpreter.outputs_size() != 2) return Status::kUnexpectedTensors;
  // The converter does not guarantee the order of the outputs, so they are told apart by their shape.
  for (size_t i = 0; i < 2; ++i) {
    TfLiteTensor* t = interpreter.output(i);
    if (isInt8Vector(t, kNumClasses)) g_probs = t;
    else if (isInt8Vector(t, kGapSize)) g_gap = t;
  }
  if (g_probs == nullptr || g_gap == nullptr) return Status::kUnexpectedTensors;
  return Status::kOk;
}

Status invoke() {
#if KWS_PROFILE
  g_profiler.reset();
#endif
  const uint32_t start = DWT->CYCCNT;
  const TfLiteStatus st = g_interpreter->Invoke();
  g_last_invoke_cycles = DWT->CYCCNT - start;
  return st == kTfLiteOk ? Status::kOk : Status::kInvokeFailed;
}

int8_t* input() { return g_input->data.int8; }
const int8_t* probabilities() { return g_probs->data.int8; }
const int8_t* gapVector() { return g_gap->data.int8; }

QuantParams inputQuant() { return quant(g_input); }
QuantParams probabilitiesQuant() { return quant(g_probs); }
QuantParams gapQuant() { return quant(g_gap); }

uint32_t schemaVersion() { return g_model->version(); }
size_t arenaCapacity() { return sizeof(g_arena); }
size_t arenaUsed() { return g_interpreter->arena_used_bytes(); }

uint32_t cycleCounter() { return DWT->CYCCNT; }
uint32_t lastInvokeCycles() { return g_last_invoke_cycles; }

#if KWS_PROFILE
int opCount() { return g_profiler.count(); }
const char* opTag(int i) { return g_profiler.tag(i); }
uint32_t opCycles(int i) { return g_profiler.cycles(i); }
#else
int opCount() { return 0; }
const char* opTag(int) { return ""; }
uint32_t opCycles(int) { return 0; }
#endif

}  // namespace kws_model
