// ram_baseline.ino — milestone M1 step 4: RAM baseline of the Arduino Mbed core on the Nano 33 BLE Sense (Lite).
//
// Purpose: measure on the board how much of the 256 KB SRAM the core + RTOS + USB serial already take and how much is
// left for the KWS firmware (tensor arena, audio ring, log-mel ring, scratch). One source file, two builds:
//   A  RAM_BASELINE_WITH_KWS_BUFFERS=0 (default) — "empty" sketch: core, RTOS, USB serial only
//   B  RAM_BASELINE_WITH_KWS_BUFFERS=1           — plus the static KWS buffers of the plan: tensor arena of
//      KWS_ARENA_KB (default 160), audio ring 16000 x int16 = 32,000 B, log-mel ring 97 x 40 float = 15,520 B
//   arduino-cli compile -b arduino:mbed_nano:nano33ble --build-property "build.extra_flags=-DRAM_BASELINE_WITH_KWS_BUFFERS=1 -DKWS_ARENA_KB=160" firmware/bringup/ram_baseline
//
// Every 2 s the sketch prints one block (a host that opens the port later still gets a complete block):
//   LNK    linker symbols __data_start__ .. __StackTop: the RAM layout as linked (sizes of .data, .bss, heap region, ISR stack)
//   SYS    CPUID, compiler, RAM/ROM regions as Mbed sees them (mbed_stats_sys_get)
//   HEAP0  heap statistics before the probe (mbed_stats_heap_get): bytes in use by the core/USB/RTOS at run time
//   THR    one line per RTOS thread: name, stack reserved / used / free (mbed_stats_thread_get_each)
//   STK    accumulated stack statistics (mbed_stats_stack_get)
//   PROBE  largest single block malloc() still hands out (descending search in 512 B steps) and the total obtainable
//          in 1 KB pieces; everything is freed again
//   HEAP   heap statistics after the probe (must equal HEAP0, otherwise the probe leaked)
//   KWS    addresses, sizes and a checksum of the static KWS buffers (build B); the checksum keeps them in the image
//   SUM    the budget line: static, main stack, core static without it, heap region, heap in use, largest free block
//
// Needs MBED_ALL_STATS_ENABLED (set in the variant's mbed_config.h of core 4.6.0; the stats then cost a little heap).
// Board: arduino:mbed_nano:nano33ble, core "Arduino Mbed OS Nano Boards" 4.6.0. No libraries needed.

#include <Arduino.h>
#include <mbed.h>
#include "platform/mbed_stats.h"
#include <stdlib.h>
#include <string.h>

#ifndef RAM_BASELINE_WITH_KWS_BUFFERS
#define RAM_BASELINE_WITH_KWS_BUFFERS 0
#endif
#ifndef KWS_ARENA_KB
#define KWS_ARENA_KB 160
#endif

extern "C" {
  // linker_script.ld of the variant
  extern char __bss_start__[], __bss_end__[], __end__[];
  extern char __HeapLimit[];   // __data_start__, __data_end__, __StackLimit, __StackTop: declared (as scalars) by the Mbed headers
  // Mbed OS main thread stack: a static array in .bss (libmbed.a, mbed_rtos_rtx.c)
  extern char _main_stack[];
}

#if RAM_BASELINE_WITH_KWS_BUFFERS
alignas(16) static uint8_t g_arena[KWS_ARENA_KB * 1024];
static int16_t g_audio_ring[16000];
static float   g_logmel_ring[97 * 40];
static const uint32_t kKwsBytes = sizeof(g_arena) + sizeof(g_audio_ring) + sizeof(g_logmel_ring);
#else
static const uint32_t kKwsBytes = 0;
#endif

static uint32_t g_iter = 0;

static uint32_t addr(const void* p) { return (uint32_t)(uintptr_t)p; }
static void kv(const char* k, uint32_t v)   { Serial.print(' '); Serial.print(k); Serial.print('='); Serial.print(v); }
static void kx(const char* k, uint32_t v)   { Serial.print(' '); Serial.print(k); Serial.print("=0x"); Serial.print(v, HEX); }
static void ks(const char* k, const char* v){ Serial.print(' '); Serial.print(k); Serial.print('='); Serial.print(v ? v : "?"); }

// Largest single block malloc() still provides, searched downwards from `cap` in 512-byte steps so that every failed
// attempt leaves the heap untouched. (A binary search from below under-reports by up to 2x: newlib's dlmalloc asks
// _sbrk for the whole new request rather than for the missing part, and does not trim a freed top chunk below 128 KB,
// so after one mid-sized success the break sits in the middle of the region and larger requests fail.) Freed again.
static uint32_t largestMalloc(uint32_t cap) {
  for (uint32_t n = cap & ~511u; n > 0; n -= 512) {
    void* p = malloc(n);
    if (p) { free(p); return n; }
  }
  return 0;
}

// Total payload obtainable in 1 KB pieces (up to 256 of them): what is really left for small allocations. All freed again.
static uint32_t fillMalloc1k(uint32_t& pieces) {
  void* blocks[256];
  pieces = 0;
  while (pieces < 256) {
    void* p = malloc(1024);
    if (!p) break;
    blocks[pieces++] = p;
  }
  for (uint32_t i = pieces; i > 0; --i) free(blocks[i - 1]);
  return pieces * 1024u;
}

static void printHeap(const char* tag, const mbed_stats_heap_t& h) {
  Serial.print(tag);
  kv("current", h.current_size); kv("max", h.max_size); kv("total", h.total_size); kv("reserved", h.reserved_size);
  kv("allocs", h.alloc_cnt); kv("fails", h.alloc_fail_cnt); kv("overhead", h.overhead_size);
  Serial.println();
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  const uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3000) {}
  Serial.print("HELLO kws-ua-nano33ble bringup ram_baseline v1.1 build=");
  Serial.print(RAM_BASELINE_WITH_KWS_BUFFERS ? "B_with_kws_buffers" : "A_empty");
  kv("arena_kb", RAM_BASELINE_WITH_KWS_BUFFERS ? KWS_ARENA_KB : 0);
  Serial.println();
}

void loop() {
  ++g_iter;
  digitalWrite(LED_BUILTIN, (g_iter & 1) ? HIGH : LOW);

  const uint32_t data_start = addr(&__data_start__), data_end = addr(&__data_end__);
  const uint32_t bss_start = addr(__bss_start__), bss_end = addr(__bss_end__), heap_start = addr(__end__);
  const uint32_t heap_limit = addr(__HeapLimit), stack_limit = addr(&__StackLimit), stack_top = addr(&__StackTop);

  Serial.print("LNK"); kv("iter", g_iter);
  kx("data_start", data_start); kx("data_end", data_end); kx("bss_start", bss_start); kx("bss_end", bss_end);
  kx("heap_start", heap_start); kx("heap_limit", heap_limit); kx("stack_limit", stack_limit); kx("stack_top", stack_top);
  kv("data", data_end - data_start); kv("bss", bss_end - bss_start); kv("static", bss_end - data_start);
  kv("heap_region", heap_limit - heap_start); kv("isr_stack", stack_top - stack_limit);
  kx("main_stack_addr", addr(_main_stack));
  Serial.println();

  mbed_stats_sys_t sys; mbed_stats_sys_get(&sys);
  Serial.print("SYS"); kx("cpuid", sys.cpu_id); kv("compiler_id", (uint32_t)sys.compiler_id); kv("compiler_version", sys.compiler_version);
  kx("ram0_start", sys.ram_start[0]); kv("ram0_size", sys.ram_size[0]); kx("rom0_start", sys.rom_start[0]); kv("rom0_size", sys.rom_size[0]);
  Serial.println();

  mbed_stats_heap_t h0; mbed_stats_heap_get(&h0);
  printHeap("HEAP0", h0);

  mbed_stats_thread_t th[10];
  const size_t nth = mbed_stats_thread_get_each(th, 10);
  uint32_t main_stack_size = 0, main_stack_used = 0;
  for (size_t i = 0; i < nth; ++i) {
    const uint32_t used = th[i].stack_size - th[i].stack_space;
    Serial.print("THR"); ks("name", th[i].name); kx("id", th[i].id); kv("state", th[i].state); kv("prio", th[i].priority);
    kv("stack", th[i].stack_size); kv("used", used); kv("free", th[i].stack_space);
    Serial.println();
    if (th[i].name && strcmp(th[i].name, "main") == 0) { main_stack_size = th[i].stack_size; main_stack_used = used; }
  }
  mbed_stats_stack_t st; mbed_stats_stack_get(&st);
  Serial.print("STK"); kv("threads", st.stack_cnt); kv("reserved", st.reserved_size); kv("max_used", st.max_size);
  Serial.println();

  const uint32_t largest = largestMalloc(h0.reserved_size - h0.current_size - h0.overhead_size);
  uint32_t pieces = 0;
  const uint32_t fill = fillMalloc1k(pieces);
  Serial.print("PROBE"); kv("largest_malloc", largest); kv("fill_1k_total", fill); kv("fill_pieces", pieces); Serial.println();

  mbed_stats_heap_t h1; mbed_stats_heap_get(&h1);
  printHeap("HEAP", h1);

#if RAM_BASELINE_WITH_KWS_BUFFERS
  g_arena[g_iter % sizeof(g_arena)] = (uint8_t)g_iter;
  g_audio_ring[g_iter % 16000] = (int16_t)g_iter;
  g_logmel_ring[g_iter % (97 * 40)] = (float)g_iter;
  uint32_t sum = 0;
  for (uint32_t i = 0; i < sizeof(g_arena); i += 1024) sum += g_arena[i];
  sum += (uint32_t)g_audio_ring[g_iter % 16000] + (uint32_t)g_logmel_ring[g_iter % (97 * 40)];
  Serial.print("KWS"); kx("arena", addr(g_arena)); kv("arena_bytes", sizeof(g_arena));
  kx("audio_ring", addr(g_audio_ring)); kv("audio_bytes", sizeof(g_audio_ring));
  kx("logmel_ring", addr(g_logmel_ring)); kv("logmel_bytes", sizeof(g_logmel_ring));
  kv("total", kKwsBytes); kv("checksum", sum);
  Serial.println();
#else
  Serial.print("KWS"); kv("total", 0); Serial.println();
#endif

  const uint32_t stat = bss_end - data_start;
  Serial.print("SUM");
  kv("ram_total", stack_top - 0x20000000u); kv("reserved_low", data_start - 0x20000000u);
  kv("static", stat); kv("kws_buffers", kKwsBytes); kv("main_stack", main_stack_size); kv("main_stack_used", main_stack_used);
  kv("core_static_wo_main_stack", stat - kKwsBytes - main_stack_size);
  kv("heap_region", heap_limit - heap_start); kv("heap_in_use", h1.current_size + h1.overhead_size);
  kv("heap_largest_free", largest); kv("heap_fill_total", fill); kv("isr_stack", stack_top - stack_limit);
  Serial.println();

  delay(2000);
}
