// selftest_model.cpp — M2 test of the model on the board (KWS_TEST_MODE 1): 25 Invoke calls on the 5 golden vectors.
//   * Runs 1-5 are test V1: the device's int8 outputs are compared with the PC (LiteRT 2.3.0, golden_set.h).
//   * Runs 6-25 are the benchmark: microseconds, CPU cycles (DWT), per-operator cycles, instruction-cache hit rate.
// The opt= field of the HELLO line shows whether this build used -Os or a speed optimisation.
// Report lines: STEP, RUN, HELLO, SYS, ALLOC, QUANT, V1, PROB8, GAP8, BENCH, OP, MEM, ERR (parsed by tools/m2_bench_parse.py).
#include "config.h"
#if KWS_TEST_MODE == 1

#include <Arduino.h>
#include <mbed.h>
#include "platform/mbed_stats.h"

#include "golden_set.h"
#include "kws_model.h"
#include "model_data.h"
#include "selftest.h"

static const int kV1Runs = GOLDEN_N_FEAT;
static const int kBenchRuns = 20;

struct V1Result {
  int top_device, top_pc, prob_max_diff, prob_diff_n, gap_max_diff, gap_diff_n;
  int8_t probs[kws_model::kNumClasses];
  int8_t gap[kws_model::kGapSize];
};

static kws_model::Status g_begin_status = kws_model::Status::kOk;
static kws_model::Status g_invoke_status = kws_model::Status::kOk;
static uint32_t g_begin_us = 0;
static V1Result g_v1[kV1Runs];
static uint32_t g_us_min = 0xFFFFFFFFu, g_us_max = 0, g_cyc_min = 0xFFFFFFFFu, g_cyc_max = 0;
static uint64_t g_us_sum = 0, g_cyc_sum = 0, g_hit_sum = 0, g_miss_sum = 0;
static uint64_t g_op_sum[kws_model::kMaxOps];
static int g_op_n = 0;
static uint32_t g_report_n = 0;
static bool g_ran = false;

static int argmaxInt8(const int8_t* v, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
  return best;
}

static void compareVectors(const int8_t* a, const int8_t* b, int n, int& max_diff, int& diff_n) {
  max_diff = 0; diff_n = 0;
  for (int i = 0; i < n; ++i) {
    const int d = abs((int)a[i] - (int)b[i]);
    if (d > max_diff) max_diff = d;
    if (d != 0) ++diff_n;
  }
}

static void printVector(const char* tag, int vec, const int8_t* v, int n) {
  Serial.print(tag); Serial.print(" vec="); Serial.print(vec);
  for (int i = 0; i < n; ++i) { Serial.print(i ? ',' : ' '); Serial.print((int)v[i]); }
  Serial.println();
}

static void printU64(uint64_t v) {
  // Print's 64-bit overload is not used here so that the output format stays plain decimal on every core version.
  char buf[21];
  int i = 20;
  buf[i] = '\0';
  do { buf[--i] = (char)('0' + (int)(v % 10)); v /= 10; } while (v != 0);
  Serial.print(&buf[i]);
}

static void printMem() {
  mbed_stats_heap_t heap; mbed_stats_heap_get(&heap);
  Serial.print("MEM heap_current="); Serial.print(heap.current_size);
  Serial.print(" heap_max="); Serial.print(heap.max_size);
  Serial.print(" heap_reserved="); Serial.print(heap.reserved_size);
  mbed_stats_thread_t th[10];
  const size_t n = mbed_stats_thread_get_each(th, 10);
  for (size_t i = 0; i < n; ++i) {
    if (th[i].name && strcmp(th[i].name, "main") == 0) {
      Serial.print(" main_stack="); Serial.print(th[i].stack_size);
      Serial.print(" main_stack_used="); Serial.print(th[i].stack_size - th[i].stack_space);
    }
  }
  Serial.println();
}

static void printQuant(const char* name, const kws_model::QuantParams& q) {
  Serial.print(' '); Serial.print(name); Serial.print("_scale="); Serial.print(q.scale, 8);
  Serial.print(' '); Serial.print(name); Serial.print("_zp="); Serial.print(q.zero_point);
}

void selftestReport() {
  ++g_report_n;
#if defined(__OPTIMIZE_SIZE__)
  const char* opt = "size";
#else
  const char* opt = "speed";
#endif
  Serial.print("HELLO kws-ua-nano33ble kws_nano33 m2-litert-micro v1 lib=85c6999 report="); Serial.print(g_report_n);
  Serial.print(" model="); Serial.print(KWS_MODEL_SHA256_8);
  Serial.print(" opt="); Serial.println(opt);
  Serial.print("SYS clock="); Serial.print(SystemCoreClock);
  Serial.print(" icache="); Serial.println((NRF_NVMC->ICACHECNF & NVMC_ICACHECNF_CACHEEN_Msk) ? 1 : 0);
  Serial.print("ALLOC status="); Serial.print((int)g_begin_status);
  Serial.print(" us="); Serial.print(g_begin_us);
  Serial.print(" arena_capacity="); Serial.print((uint32_t)kws_model::arenaCapacity());
  Serial.print(" arena_used="); Serial.println((uint32_t)kws_model::arenaUsed());
  if (!g_ran) {
    Serial.println("ERR model not run, see the ALLOC status and the runtime messages above");
    return;
  }
  Serial.print("QUANT");
  printQuant("in", kws_model::inputQuant());
  printQuant("prob", kws_model::probabilitiesQuant());
  printQuant("gap", kws_model::gapQuant());
  Serial.println();
  for (int i = 0; i < kV1Runs; ++i) {
    const V1Result& r = g_v1[i];
    Serial.print("V1 vec="); Serial.print(i);
    Serial.print(" label="); Serial.print((int)GOLDEN_LABELS[i]);
    Serial.print(" top_device="); Serial.print(r.top_device);
    Serial.print(" top_pc="); Serial.print(r.top_pc);
    Serial.print(" prob_max_diff="); Serial.print(r.prob_max_diff);
    Serial.print(" prob_diff_n="); Serial.print(r.prob_diff_n);
    Serial.print(" gap_max_diff="); Serial.print(r.gap_max_diff);
    Serial.print(" gap_diff_n="); Serial.println(r.gap_diff_n);
    printVector("PROB8", i, r.probs, kws_model::kNumClasses);
    printVector("GAP8", i, r.gap, kws_model::kGapSize);
  }
  const uint64_t cyc_mean = g_cyc_sum / kBenchRuns;
  Serial.print("BENCH runs="); Serial.print(kBenchRuns);
  Serial.print(" status="); Serial.print((int)g_invoke_status);
  Serial.print(" us_min="); Serial.print(g_us_min);
  Serial.print(" us_mean="); printU64(g_us_sum / kBenchRuns);
  Serial.print(" us_max="); Serial.print(g_us_max);
  Serial.print(" cycles_min="); Serial.print(g_cyc_min);
  Serial.print(" cycles_mean="); printU64(cyc_mean);
  Serial.print(" cycles_max="); Serial.print(g_cyc_max);
  Serial.print(" icache_hit="); printU64(g_hit_sum / kBenchRuns);
  Serial.print(" icache_miss="); printU64(g_miss_sum / kBenchRuns);
  Serial.println();
  for (int i = 0; i < g_op_n; ++i) {
    const uint64_t c = g_op_sum[i] / kBenchRuns;
    Serial.print("OP i="); Serial.print(i);
    Serial.print(" tag="); Serial.print(kws_model::opTag(i));
    Serial.print(" cycles_mean="); printU64(c);
    Serial.println();
  }
  printMem();
}

static bool runOnce(int run) {
  const int vec = run % GOLDEN_N_FEAT;
  memcpy(kws_model::input(), GOLDEN_INPUT[vec], kws_model::kInputBytes);
  NRF_NVMC->IHIT = 0;
  NRF_NVMC->IMISS = 0;
  const uint32_t t = micros();
  g_invoke_status = kws_model::invoke();
  const uint32_t us = micros() - t;
  const uint32_t hit = NRF_NVMC->IHIT, miss = NRF_NVMC->IMISS;
  const uint32_t cyc = kws_model::lastInvokeCycles();

  Serial.print("RUN n="); Serial.print(run);
  Serial.print(" vec="); Serial.print(vec);
  Serial.print(" status="); Serial.print((int)g_invoke_status);
  Serial.print(" us="); Serial.print(us);
  Serial.print(" cycles="); Serial.println(cyc);
  if (g_invoke_status != kws_model::Status::kOk) return false;

  if (run < kV1Runs) {
    V1Result& r = g_v1[vec];
    memcpy(r.probs, kws_model::probabilities(), kws_model::kNumClasses);
    memcpy(r.gap, kws_model::gapVector(), kws_model::kGapSize);
    compareVectors(r.probs, GOLDEN_PROB[vec], kws_model::kNumClasses, r.prob_max_diff, r.prob_diff_n);
    compareVectors(r.gap, GOLDEN_GAP[vec], kws_model::kGapSize, r.gap_max_diff, r.gap_diff_n);
    r.top_device = argmaxInt8(r.probs, kws_model::kNumClasses);
    r.top_pc = argmaxInt8(GOLDEN_PROB[vec], kws_model::kNumClasses);
  } else {
    if (us < g_us_min) g_us_min = us;
    if (us > g_us_max) g_us_max = us;
    if (cyc < g_cyc_min) g_cyc_min = cyc;
    if (cyc > g_cyc_max) g_cyc_max = cyc;
    g_us_sum += us;
    g_cyc_sum += cyc;
    g_hit_sum += hit;
    g_miss_sum += miss;
    g_op_n = kws_model::opCount();
    for (int i = 0; i < g_op_n && i < kws_model::kMaxOps; ++i) g_op_sum[i] += kws_model::opCycles(i);
  }
  return true;
}

void selftestRun(kws_model::Status begin_status, uint32_t begin_us) {
  g_begin_status = begin_status;
  g_begin_us = begin_us;
  if (begin_status != kws_model::Status::kOk) return;

  NRF_NVMC->ICACHECNF |= NVMC_ICACHECNF_CACHEPROFEN_Msk;  // keep the cache on, add its hit and miss counters

  selftestStep("runs");
  for (int run = 0; run < kV1Runs + kBenchRuns; ++run) {
    if (!runOnce(run)) return;
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }
  g_ran = true;
  selftestStep("done");
}

#endif  // KWS_TEST_MODE == 1
