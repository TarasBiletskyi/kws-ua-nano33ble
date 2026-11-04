# M1 — board bring-up results

Board: Arduino Nano 33 BLE Sense **Lite** (Tiny Machine Learning Kit; u-blox NINA-B306 = nRF52840), USB VID:PID 2341:805A,
port **COM4** on the development PC (Windows 11). Toolchain: `arduino-cli` 1.5.1, core `arduino:mbed_nano` 4.6.0
(ArduinoCore-mbed 4.6.0), `arm-none-eabi-gcc` 7-2017q4, bossac 1.9.1-arduino2.

## Step 1 (2026-11-01) — toolchain

Installed `arduino-cli` 1.5.1, Arduino IDE 2.3.10 and the core `arduino:mbed_nano` 4.6.0 (≈720 MB in `Arduino15`).
`arduino-cli board list` → `COM4  Arduino Nano 33 BLE  arduino:mbed_nano:nano33ble` (the Sense / Sense Lite have no
separate board entry).

## Step 2 (2026-11-01) — first firmware: `firmware/bringup/blink_hello`

| Item | Result |
|---|---|
| Compile (`arduino-cli compile -b arduino:mbed_nano:nano33ble`) | OK, 13 s for the first build; one harmless core warning (`No configuration for LF clock source`, Xtal used) |
| Flash used by the sketch (`.bin`) | **85,712 B** = 8 % of 983,040 B (bootloader excluded; `with_bootloader.bin` is 151,256 B) |
| Static RAM (`Global variables`) | **44,552 B** = 16 % of 262,144 B → 217,592 B left for heap and stacks *before* the Mbed OS thread stacks are carved out (measured at step 4) |
| Upload (`arduino-cli upload -p COM4`, bossac via the board bootloader) | OK, 9.0 s end to end, 21 × 4 KB pages written in 3.6 s; the board resets itself; the port stays COM4 |
| LED | three 80 ms flashes + pause, 1 s period (visual check by the user) |
| USB serial, 115200 baud, host asserts DTR | 13 heartbeat lines in 12 s, period 1.000 ± 0.001 s, counter continuous (n = 21…33, uptime 24.0…36.0 s), no garbage, no loss |

Baseline for the RAM budget: an almost empty Mbed sketch already costs ≈44.6 KB of static RAM and ≈85.7 KB of flash.
The KWS budget from the plan (≈135–145 KB tensor arena + 32 KB audio ring + ≈16 KB log-mel ring + scratch) has to fit
into what remains after the RTOS stacks, which step 4 measures on the board.

Notes: nothing else on the PC holds the port (the IDE has never been started); the bootloader is entered automatically
by the 1200-baud touch. If an upload ever fails, double-tap RESET (the orange LED then pulses) and repeat the command.

## Step 3 (2026-11-01/02) — PDM microphone: `firmware/bringup/pdm_rms`

Driver facts (core 4.6.0, `libraries/PDM/src/nrf52/PDM.cpp`): 16 kHz = PDM clock 1.28 MHz with ratio 80, mono,
left-falling edge; the gain goes straight into the nRF52840 GAIN register (0 = −20 dB, 20 = −10 dB **driver default**,
40 = 0 dB, 80 = +20 dB, 0.5 dB per step); 512-byte double buffer, i.e. 256 samples = 16 ms per callback, and the callback
runs inside the PDM IRQ (priority 7). If the application has not consumed a buffer before the next one is full, the
driver calls `nrf_pdm_disable()` and the stream stops silently (no flag), so the firmware has to watch the callback
counter and restart. The driver does not remove DC.

Sketch: reads every buffer inside the callback, accumulates sum / sum of squares / peak, prints one `RMS` line per
100 ms (`ac` = DC-free RMS in dBFS, `pk` = peak), a `SUM` line every 60 s, commands `G <0..80>` / `S` / `Z`, stall
watchdog with automatic restart. Flash 92,464 B (9 %), static RAM 45,312 B (17 %). PC side: `tools/serial_log.ps1`
(logging with scheduled commands, no pyserial needed) and `tools/pdm_analyze.py`.

### 3a — stream integrity

Every 100 ms interval carried 1,536 or 1,792 samples (6 or 7 callbacks), mean 1,600 = exactly 16 kHz; 0 short reads and
0 stalls in all short runs (12 s, 70 s, 25 s, 20 s). The 10-minute run is in 3f.

### 3b — start-up transient

After `PDM.begin()` the DC offset starts near −70 counts and decays over ≈2 s (−68 → −10 counts within 1.3 s); the AC
floor is valid from the first callback. The loudest "event" of every run (`max_pk` ≈ −28 dBFS at gain 20) falls into the
first second after start. **Decision:** the KWS firmware discards the first 2 s after `PDM.begin()` (the plan had 10
frames = 160 ms) and keeps the optional DC blocker.

### 3c — noise floor vs gain (board next to the laptop, quiet room, 15 s per gain)

| gain reg | gain | ac median dBFS | ac p5 | ac min | pk max |
|---|---|---|---|---|---|
| 20 | −10 dB | −83.5 | −86.9 | −87.6 | −64.7 |
| 40 | 0 dB | −73.1 | −77.0 | −78.0 | −49.2 |
| 60 | +10 dB | −62.9 | −65.6 | −66.6 | −45.3 |
| 80 | +20 dB | −52.3 | −55.8 | −56.7 | −34.2 |

The floor rises 10 dB per 10 dB of gain, so it is acoustic (room + laptop fan + microphone self-noise), not digital.
≈ −73 dBFS at 0 dB gain corresponds to ≈47 dB SPL with the datasheet sensitivity of −26 dBFS at 94 dB SPL. At the chosen
+20 dB the floor (−52 dBFS) matches the "−50 dBFS" row of the PC emulation `noise_floor_check.md` (78–81 % accuracy
without, 88–91 % with the 6 dB frame gate) — **the frame gate is mandatory**, now confirmed with the real microphone.
Log `tools/out/pdm/sweep_near_laptop.log` (untracked), JSON `m1_pdm_sweep_near_laptop.json`.

### 3d — speech level at +20 dB: 13 keywords, one speaker, ≈0.5 m, normal voice

Speech-active 100 ms intervals (ac > −32 dBFS): mean −28 dBFS, max −16 dBFS; typical peaks −18 dBFS, loudest −10.2 dBFS
(coinciding with a DC excursion of −1,058 counts = a low-frequency thump); floor between words −50.5 dBFS (median).
Speech-to-floor ≈22–26 dB, headroom to clipping ≥10 dB. The corpus clips sit at ≈ −30 dBFS, so +20 dB puts the device
at the training level. **Decision: default gain register 80 (+20 dB); fallback 70 (+15 dB) if live tests show peaks
near 0 dBFS.** JSON `m1_pdm_speech_gain80.json`.

### 3e — board moved away from the laptop (+20 dB, 20 s)

First attempt (20 s): a steady −43 dBFS for 17.5 s, then −50…−60 dBFS with a minimum of −60.6 dBFS (near the laptop the
minimum was −56.7). Repeat (first 32 s of the stability run, same position): a steady −44 dBFS (median −43.9, p5 −45.2)
for the whole 30 s with a few spikes to −34 dBFS. So the new position is **8 dB noisier** than next to the laptop
(−52 dBFS), not quieter: the floor is dominated by a steady local source at that spot (airflow or vibration coupling —
to be identified; the laptop's own fan contributes at most ≈4 dB judging by the quiet tail of the first attempt).
Conclusion: the floor is position-dependent by ≈10 dB within one desk; for the demo the board goes to the quietest
spot, and the firmware's frame gate must estimate the floor adaptively rather than use a constant. JSON
`m1_pdm_quiet_far_gain80.json`, `m1_pdm_quiet_far_repeat_gain80.json`.

### 3f — 10-minute stability run (+20 dB, board away from the laptop, normal room activity)

| Item | Result |
|---|---|
| Duration / lines | 600.1 s, 6,001 `RMS` lines, 11 `SUM` lines |
| Samples | 9,601,536 received vs 9,601,600 expected from the timestamps (**0.001 %**, the last partial interval); on-board `SUM`: 9,580,544 vs 9,580,768 (0.002 %) |
| Callbacks | 37,424; every 100 ms interval had 6 (4,501×) or 7 (1,500×) callbacks of 256 samples = exactly 16 kHz on average |
| Short reads / stalls / host-side gaps > 250 ms | 0 / 0 / 0 |
| Floor over 10 min (DC-free RMS) | median −52.2 dBFS, p5 −54.5, min −56.6; loudest peak −8.8 dBFS (room activity) |

**M1 gate "PDM 16 kHz stable for 10 minutes without overflow": passed.** Caveat: this sketch does about a microsecond
of work per sample inside the callback; the KWS firmware computes the log-mel per 10 ms frame in the main loop while
`Invoke()` blocks for hundreds of milliseconds, so there the 1-s ring buffer and the catch-up logic of the plan carry
this property — to be re-checked at M4. JSON `m1_pdm_stability_gain80.json`.

## Step 4 (2026-11-03) — RAM baseline: `firmware/bringup/ram_baseline`

Question: how much of the 256 KB SRAM do the Mbed core, the RTOS and the USB serial link take, and does the plan's KWS
budget (tensor arena 160 KB + audio ring 32,000 B + log-mel ring 15,520 B = 211,360 B) fit next to them? One sketch,
two builds (`-DRAM_BASELINE_WITH_KWS_BUFFERS=1 -DKWS_ARENA_KB=160` for build B), printing every 2 s the linker symbols,
the Mbed heap statistics (`mbed_stats_heap_get`), the per-thread stack statistics (`mbed_stats_thread_get_each`), a
malloc probe and the addresses of the KWS buffers. Logs `tools/out/ram/ram_baseline_{a,b}.log` (untracked), JSON
`m1_ram_baseline.json` (`tools/ram_baseline_parse.py`). Two flashes, both boards talked back within 1.6 s.

### 4a — RAM map as linked, and what the board reports

RAM 0x20000000–0x20040000 (262,144 B). The first 0x200 = 512 B are reserved by the core (RAM vector table 0x100 + crash
data 0x100), the last 0x400 = 1,024 B are the ISR/boot stack (`__StackLimit`…`__StackTop`). Everything else, 260,608 B,
is `.data` + `.bss` + heap (`__end__`…`__HeapLimit`); the heap takes whatever the statics leave.

| | build A (empty) | build B (+ KWS buffers) |
|---|---|---|
| Flash (`Sketch uses`) | 88,272 B (8 %) | 88,712 B (9 %) |
| Static = `.data` + `.bss` (compiler: `Global variables`) | 3,560 + 40,992 = **44,552 B** (16 %) | 3,560 + 252,352 = **255,912 B** (97 %); 255,920 B by the linker symbols, the `alignas(16)` arena shifts `.bss` by 8 B |
| of which Mbed main-thread stack `_main_stack` (a static array in `.bss`) | 32,768 B | 32,768 B |
| Core static without the main stack | **11,784 B** | 11,792 B |
| of which KWS buffers | — | 211,360 B: arena 163,840 @ 0x2000C9B0, audio ring 32,000 @ 0x2000100C, log-mel ring 15,520 @ 0x20008D0C |
| Heap region (`__end__`…`__HeapLimit`) | **216,056 B** | **4,688 B** |
| Heap in use at run time (`current` + statistics overhead) | **976 B**, 7 allocations, constant over the run | 976 B, 7 allocations |
| Heap peak since reset (`max`) | polluted by the v1 probe | 1,388 B = steady state + the 512 B probe → no start-up transient above ≈1.4 KB |
| RTOS threads: reserved / max used | main 32,768 / 824; rtx_idle 512 / 392; rtx_timer 768 / 104; USBevt 256 / 136 (heap-allocated) | main 32,768 / 1,912 (the probe's 1 KB pointer array); the others identical |
| `mbed_stats_stack_get`: stacks, reserved, max used | 4, 34,304 B, 1,456 B | 4, 34,304 B, 2,544 B |
| Largest single `malloc()` | 113,671 B (sketch v1, binary search — an artefact, see 4c) | 512 B (sketch v1.1, descending search — see 4c) |
| Total obtainable in 1 KB pieces | — (v1 had no fill probe) | 0 B |
| Boots, USB serial, LED | yes; 72 lines / 12 s | **yes**; 72 lines / 12 s, checksum over the buffers advancing, heap figures constant |

### 4b — where the 44.5 KB of static RAM go (build A, `arm-none-eabi-nm -S --size-sort`)

| Symbols | Bytes | Note |
|---|---|---|
| `_main_stack` | 32,768 | Mbed OS main thread = the thread that runs `setup()`/`loop()`. The empty sketch uses 824 B of it, the probe build 1,912 B, so ≈30 KB of it are **ours** for per-window scratch (FFT buffers, MFCC rows) |
| `PluggableUSBD()::obj` 1,424, `_SerialUSB` 696, `m_ep_state` 288, `_UART_USB_` 308, `_UART1_` 308 | 3,024 | USB CDC device, Arduino `Serial` objects, Mbed UART handles |
| `impure_data` 1,064, `__malloc_av_` 1,032, `__global_locale` 364, `filehandles` 256 | 2,716 | newlib: stdio state, malloc bins, locale, file handles |
| `os_timer_thread_stack` 768, `os_idle_thread_stack` 512 (+ control blocks below 256 B) | 1,280 | RTX5 kernel threads — static, already inside the 44.5 KB |
| `g_APinDescription` | 544 | Arduino pin table |
| `bbEncryptTxBuf` 256, `bbEncryptOutBuf` 260, `bbEncryptScratchBuf` 269 | 785 | BLE link-layer crypto buffers the linker keeps although BLE is unused |
| 166 symbols below 256 B | 3,295 | drivers, RTOS objects, USB endpoint state, our sketch |

(The RAM vector table `nrf_dispatch_vector`, 256 B at 0x20000000, lives in the reserved low 512 B, not in `.bss`; the
remaining ≈140 B of the 44,552 are alignment padding.)

### 4c — the two malloc probes, and what they say about the heap

Build A, sketch v1 (binary search from below): after the first success (108,028 B) newlib's dlmalloc keeps the freed block
as the top chunk (it trims only above 128 KB) and, for any request larger than that chunk, asks `_sbrk` for the **whole**
new size rather than for the missing part, so every request above the top-chunk size (≈113.7 KB) fails although ≈215 KB
are free. The 113,671 B figure is the size of that top chunk, not the capacity; the capacity of build A is heap region −
in use = 216,056 − 976 ≈ 215 KB. Sketch v1.1 searches downwards (every failed attempt leaves the heap untouched) and adds
a 1 KB fill probe.

Build B, sketch v1.1: of the 4,688 B heap region only 512 B could be allocated and the 1 KB fill got nothing. Reason:
dlmalloc extends the heap in 4,096 B pages and at start rounds the break up to the next page boundary (0x2003F000), which
leaves a first chunk of 1,616 B (0x2003E9B0…0x2003F000) holding the 976 B the core needs and ≈600 B free; the remaining
3,072 B up to `__HeapLimit` can never be reached because the next extension would need a full 4,096 B page. **Rule for
the KWS firmware: leave a heap region of at least 8 KB** (one full page after the first page boundary plus the core's
1 KB) — i.e. keep the static total below ≈252 KB.

### 4d — budget and decision

| Item | Bytes |
|---|---|
| RAM for `.data` + `.bss` + heap (262,144 − 512 − 1,024) | 260,608 |
| − core static including the 32 KB main stack (measured) | −44,552 |
| = available for KWS statics + heap | 216,056 |
| − heap region (rule ≥ 8 KB; the core needs 976 B) | −8,192 |
| = **safe static budget for arena + rings + modules** | **≈207,800** |
| Plan as written: arena 160 KB + audio ring 32,000 + log-mel ring 15,520 | 211,360 → boots (build B) with a 4.7 KB heap region: **at the edge** |
| Plan with the arena at the realistic 135–145 KB (M2 measures `arena_used_bytes()`) | 185,000–195,000 → **15–23 KB spare** |

**Decision: the full model goes ahead; no stride-(4,2) variant is needed for RAM.** Firmware rules derived here:
(1) `KWS_ARENA_BYTES` starts at 160 KB only for the M2 measurement and is then set to `arena_used_bytes()` + 2 KB, with
the static total kept ≤ 252 KB; (2) the heap region stays ≥ 8 KB; (3) per-window scratch (FFT, MFCC rows) lives on the
32 KB main-thread stack, not in `.bss`; (4) the log-mel ring moves to int16 Q8.8 (−7.8 KB) only if M2 shows the arena
above ≈150 KB. The M2 gate is therefore about latency (window finish + Invoke ≤ 300 ms), not about RAM.

**M1 complete** — gates: PDM 16 kHz stable for 10 minutes (step 3), RAM baseline (this step).
