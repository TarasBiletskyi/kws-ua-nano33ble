// pdm_rms.ino — milestone M1 step 3: PDM microphone bring-up on the Arduino Nano 33 BLE Sense (Lite).
//
// What it does
//   * Starts the on-board PDM microphone (MP34DT05) at 16 kHz mono through the Mbed core's PDM library.
//   * The PDM callback (runs in the PDM IRQ) reads every 512-byte buffer (256 samples = 16 ms) and accumulates
//     sum, sum of squares, peak and sample count. The first 10 callbacks after (re)start are discarded.
//   * Every 100 ms loop() prints one line:
//       RMS t=<ms> g=<gain> n=<samples> cb=<callbacks> rms=<dBFS> ac=<dBFS> pk=<dBFS> dc=<counts> short=<k>
//     ac = RMS with the DC offset removed (the noise-floor figure), pk = peak |sample| in dBFS.
//   * The Mbed driver silently disables the PDM peripheral when a buffer is not consumed in time
//     (PDM.cpp: "buffer overflow, stop"). A watchdog detects the missing callbacks, prints STALL and restarts PDM.
//   * Serial commands (115200 baud, newline-terminated):
//       G <0..80>  set the PDM gain register (0 = -20 dB, 20 = -10 dB (driver default), 40 = 0 dB, 80 = +20 dB; 0.5 dB/step)
//       S          print a SUM line: run time, samples vs expected, loss %, callbacks, short reads, stalls, min ac, max pk
//       Z          zero the run statistics (after changing the gain or the position of the board)
//     A SUM line is also printed automatically every 60 s. LED_BUILTIN toggles once per second as an alive sign.
//
// Board: arduino:mbed_nano:nano33ble, core "Arduino Mbed OS Nano Boards" 4.6.0. Library: PDM (bundled with the core).

#include <Arduino.h>
#include <PDM.h>
#include <math.h>

static const int      kSampleRate    = 16000;
static const int      kBufBytes      = 512;   // driver default: 256 int16 samples per callback (16 ms)
static const uint32_t kReportMs      = 100;
static const uint32_t kSummaryMs     = 60000;
static const uint32_t kSkipCallbacks = 10;    // 160 ms of start-up transient discarded after every (re)start
static const uint32_t kStallMs       = 200;   // no callback for this long -> the driver has disabled PDM

static int16_t g_buf[kBufBytes / 2];

// Written in the PDM ISR, read in loop() with the PDM IRQ masked.
static volatile uint32_t g_cb_total = 0;      // never reset: stall watchdog
static volatile uint32_t g_cb_int = 0;        // per report interval
static volatile uint32_t g_samples = 0;
static volatile uint32_t g_short = 0;
static volatile uint32_t g_skip = kSkipCallbacks;
static volatile uint64_t g_sumsq = 0;
static volatile int64_t  g_sum = 0;
static volatile int32_t  g_peak = 0;

static int      g_gain = 20;                  // register value; driver default
static uint32_t g_run_start_ms = 0, g_run_samples = 0, g_run_cb = 0, g_run_short = 0, g_run_skipped = 0;
static uint32_t g_stalls = 0;
static uint32_t g_last_report = 0, g_last_summary = 0, g_last_led = 0;
static uint32_t g_wd_cb = 0, g_wd_ms = 0;
static float    g_min_ac = 0.0f, g_max_pk = -200.0f;
static bool     g_have_stats = false;
static char     g_cmd[32];
static size_t   g_cmd_len = 0;

static void onPDM() {
  const int n = PDM.read(g_buf, kBufBytes);
  g_cb_total++;
  if (g_skip) { g_skip--; return; }
  g_cb_int++;
  if (n != kBufBytes) g_short++;
  const int ns = n / 2;
  uint64_t ss = 0;
  int64_t  s = 0;
  int32_t  pk = 0;
  for (int i = 0; i < ns; ++i) {
    const int32_t x = g_buf[i];
    ss += (uint64_t)(x * x);
    s += x;
    const int32_t a = x < 0 ? -x : x;
    if (a > pk) pk = a;
  }
  g_sumsq += ss;
  g_sum += s;
  g_samples += (uint32_t)ns;
  if (pk > g_peak) g_peak = pk;
}

static float dbfs(double amp) { return amp > 0.0 ? (float)(20.0 * log10(amp / 32768.0)) : -200.0f; }

static bool pdmStart() {
  PDM.setBufferSize(kBufBytes);
  PDM.onReceive(onPDM);
  PDM.setGain(g_gain);                 // sets _gain before begin(), begin() keeps it
  g_skip = kSkipCallbacks;
  g_run_skipped += kSkipCallbacks * (kBufBytes / 2);
  if (!PDM.begin(1, kSampleRate)) return false;
  PDM.setGain(g_gain);                 // and write the register again now that the peripheral is configured
  return true;
}

static void zeroRunStats(uint32_t now) {
  g_run_start_ms = now;
  g_run_samples = g_run_cb = g_run_short = g_run_skipped = 0;
  g_min_ac = 0.0f; g_max_pk = -200.0f; g_have_stats = false;
}

static void printSummary(uint32_t now) {
  const uint32_t run_ms = now - g_run_start_ms;
  const double expected = (double)run_ms * (kSampleRate / 1000.0) - (double)g_run_skipped;
  const double loss_pct = expected > 0 ? 100.0 * (expected - (double)g_run_samples) / expected : 0.0;
  Serial.print("SUM t="); Serial.print(now);
  Serial.print(" run_s="); Serial.print(run_ms / 1000.0, 1);
  Serial.print(" samples="); Serial.print(g_run_samples);
  Serial.print(" expected="); Serial.print((uint32_t)(expected > 0 ? expected : 0));
  Serial.print(" loss_pct="); Serial.print(loss_pct, 3);
  Serial.print(" cb="); Serial.print(g_run_cb);
  Serial.print(" short="); Serial.print(g_run_short);
  Serial.print(" stalls="); Serial.print(g_stalls);
  Serial.print(" min_ac="); Serial.print(g_have_stats ? g_min_ac : -200.0f, 1);
  Serial.print(" max_pk="); Serial.print(g_have_stats ? g_max_pk : -200.0f, 1);
  Serial.print(" gain="); Serial.println(g_gain);
}

static void handleCommand(const char* cmd, uint32_t now) {
  if (cmd[0] == 'G' || cmd[0] == 'g') {
    int v = atoi(cmd + 1);
    if (v < 0) v = 0;
    if (v > 80) v = 80;
    g_gain = v;
    PDM.setGain(g_gain);
    zeroRunStats(now);
    Serial.print("OK gain="); Serial.print(g_gain);
    Serial.print(" db="); Serial.println((g_gain - 40) * 0.5, 1);
  } else if (cmd[0] == 'S' || cmd[0] == 's') {
    printSummary(now);
  } else if (cmd[0] == 'Z' || cmd[0] == 'z') {
    zeroRunStats(now);
    Serial.println("OK zeroed");
  } else if (cmd[0] != '\0') {
    Serial.print("ERR unknown command: "); Serial.println(cmd);
  }
}

static void pollSerial(uint32_t now) {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      g_cmd[g_cmd_len] = '\0';
      if (g_cmd_len > 0) handleCommand(g_cmd, now);
      g_cmd_len = 0;
    } else if (g_cmd_len < sizeof(g_cmd) - 1) {
      g_cmd[g_cmd_len++] = c;
    }
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  const uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3000) {}
  Serial.println("HELLO kws-ua-nano33ble bringup pdm_rms v1.1 sr=16000 buf=512 report_ms=100");
  zeroRunStats(millis());              // before pdmStart(): it books the discarded start-up callbacks in g_run_skipped
  const bool ok = pdmStart();
  Serial.print(ok ? "OK PDM started gain=" : "ERR PDM.begin failed gain="); Serial.println(g_gain);
  const uint32_t now = millis();
  g_last_report = g_last_summary = g_last_led = g_wd_ms = now;
}

void loop() {
  const uint32_t now = millis();
  pollSerial(now);

  // stall watchdog
  const uint32_t cb = g_cb_total;
  if (cb != g_wd_cb) { g_wd_cb = cb; g_wd_ms = now; }
  else if (now - g_wd_ms > kStallMs) {
    g_stalls++;
    Serial.print("STALL t="); Serial.print(now); Serial.print(" stalls="); Serial.print(g_stalls); Serial.println(" restarting PDM");
    PDM.end();
    if (!pdmStart()) Serial.println("ERR PDM.begin failed after stall");
    g_wd_ms = now;
  }

  if (now - g_last_report >= kReportMs) {
    g_last_report += kReportMs;
    NVIC_DisableIRQ(PDM_IRQn);
    const uint32_t n = g_samples, cbs = g_cb_int, sh = g_short;
    const uint64_t ss = g_sumsq;
    const int64_t  s = g_sum;
    const int32_t  pk = g_peak;
    g_samples = 0; g_cb_int = 0; g_short = 0; g_sumsq = 0; g_sum = 0; g_peak = 0;
    NVIC_EnableIRQ(PDM_IRQn);

    float rms = -200.0f, ac = -200.0f, pkdb = -200.0f;
    double dc = 0.0;
    if (n > 0) {
      const double mean = (double)s / (double)n;
      const double ms = (double)ss / (double)n;
      const double var = ms - mean * mean;
      rms = dbfs(sqrt(ms));
      ac = dbfs(var > 0.0 ? sqrt(var) : 0.0);
      pkdb = dbfs((double)pk);
      dc = mean;
      g_run_samples += n; g_run_cb += cbs; g_run_short += sh;
      if (!g_have_stats || ac < g_min_ac) g_min_ac = ac;
      if (!g_have_stats || pkdb > g_max_pk) g_max_pk = pkdb;
      g_have_stats = true;
    }
    Serial.print("RMS t="); Serial.print(now);
    Serial.print(" g="); Serial.print(g_gain);
    Serial.print(" n="); Serial.print(n);
    Serial.print(" cb="); Serial.print(cbs);
    Serial.print(" rms="); Serial.print(rms, 1);
    Serial.print(" ac="); Serial.print(ac, 1);
    Serial.print(" pk="); Serial.print(pkdb, 1);
    Serial.print(" dc="); Serial.print(dc, 0);
    Serial.print(" short="); Serial.println(sh);
  }

  if (now - g_last_summary >= kSummaryMs) { g_last_summary = now; printSummary(now); }
  if (now - g_last_led >= 1000) { g_last_led = now; digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); }
}
