# report/results — verification, simulation and board bring-up outputs

| File | Produced by | Content |
|---|---|---|
| `v0_two_output_equality.json` | `tools/convert_two_output.py` | V0: two-output INT8 model vs the published model on the 1,204 test clips (softmax identical, GAP identical to the published MEAN tensor), quantization parameters, model sha256, submodule commit |
| `operating_point_sim.{csv,json,md}` | `tools/simulate_operating_point.py` | device-pipeline run over the saved isolated-clip stream (`warden_stream_v2.wav`, 602 s, 153 events): every decision configuration (hop × K × threshold × suppression × gate) with Warden metrics; recommended operating points; per-hop raw probabilities are cached in `tools/out/opsim/` (not tracked) |
| `noise_floor_check.{json,md}` | `tools/noise_floor_check.py` | accuracy of the device pipeline when white/pink noise at −60/−50/−40 dBFS replaces the digital-silence padding of the test clips, with and without the frame-gate mitigation |
| `m1_bringup.md` | M1 steps, written by hand from the arduino-cli / serial output | board bring-up: toolchain versions, first flash of `firmware/bringup/blink_hello` (upload time, serial heartbeat check), PDM microphone results (noise floor vs gain, speech level, stability), RAM baseline (static / RTOS / heap of the empty core and with the plan's KWS buffers, static-RAM decomposition by symbol, the dlmalloc page-granularity finding, budget and decision) |
| `m1_pdm_*.json` | `tools/pdm_analyze.py` on `tools/out/pdm/*.log` | per-gain noise floor (p5/median/min of DC-free RMS, max peak), speech level at +20 dB, far-from-laptop check, 10-minute continuity |
| `m2_litert_micro.md` | M2 steps, written by hand from the arduino-cli / serial output | LiteRT for Microcontrollers on the board: library version, kernels, flash and RAM of the build, budget consequence, the Windows path-length pitfall, measured arena, `Invoke` at `-Os` and `-O3`, per-operator profile, V1 and the dense-layer scale defect of the old library, its replacement by a library generated from current LiteRT Micro, options for the gate |
| `m3_frontend.md` | M3 steps, written by hand | `frontend.c`: size, tests, V3 on the PC, the digital-zero case |
| `v3_frontend_pc.json` | `tools/frontend_pc_eval.py` | V3: accuracy, per-class recall and confusion matrix of the C frontend with the int8 model on 1,204 clips, differences from the NumPy and librosa paths |
| `m3_board_frontend.json` | `tools/m3_board_compare.py` on `tools/out/m3/frontend_board.log` | V2 on 3 built-in clips: features of the board against the same code on the PC and the references, frontend and `Invoke` timing |
| `device_v1.json`, `device_v2.json`, `device_v4.json` | `tools/serial_harness.py` | board results over serial: V1 (features into the model) and V2 (audio through the frontend) on 50 clips of 15 classes, V4 (whole test set) |
| `m2_bench.json` | `tools/m2_bench_parse.py` on `tools/out/m2/bench_*.log` | benchmark, cycles per operator group, cache hit rate and V1 against both PC runtimes: old library at `-Os` and `-O3`, LiteRT Micro at `-O3` |
| `m2_fc_scale_check.json` | `tools/fc_scale_check.py` | PC simulation over the 1,204 test clips: accuracy and per-class recall with the correct per-class scales and with the board's single scale |
| `m1_ram_baseline.json` | `tools/ram_baseline_parse.py` on `tools/out/ram/ram_baseline_{a,b}.log` | linker map, heap and thread statistics, malloc probes and the budget line for the empty build (A) and the build with the KWS buffers (B), plus the derived static budget |

Device-side results (V1–V7, measurements) will be added here by `tools/serial_harness.py` and `tools/collect_measurements.py`.
