# tools/ — PC side

Planned scripts (see the project plan; names fixed, implementation follows milestone by milestone):

| Script | Purpose |
|---|---|
| `export_tables.py` | window, sparse mel filterbank, DCT matrix, Savitzky–Golay kernels and edge sets, quantization parameters → `firmware/kws_nano33/fe_tables.h` |
| `convert_two_output.py` | two-output INT8 model (softmax + GAP vector) from the Keras checkpoint, equality check against the published model (runs in the model repo's TensorFlow environment) |
| `export_model.py` | `.tflite` → `model_data.cpp` (aligned C array) + `model_meta.json` |
| `export_golden.py` | golden vectors (audio, log-mel, features, int8 input, int8 output, GAP) for 50 test clips + `golden_set.h` |
| `build_frontend_pc.py`, `frontend_pc_eval.py` | compile `frontend.c` for the PC and evaluate the device arithmetic on the full test set |
| `serial_harness.py` | drive the board over USB serial: golden tests, WAV clips, whole test set, benchmarks |
| `simulate_operating_point.py` | choose hop / K / threshold / suppression on the saved test stream |
| `personalize_sim.py`, `live_test_logger.py`, `collect_measurements.py` | stage 2 and reporting |

Environment: Python 3.11, `pip install -r requirements.txt`. The scripts import `src.features` and `config.yaml` from
`external/kws-model-evaluation` (initialise with `git submodule update --init`).

## Implemented (M0, step 2 — 2026-10-31)

| Script / module | Output |
|---|---|
| `kws_paths.py` | shared paths, config, corpus/split helpers (no TensorFlow) |
| `frontend_ref.py` | table-driven NumPy frontend = executable specification of `frontend.c`; matches `src.features.extract_mfcc` to ≈6e-5 on real clips |
| `export_tables.py` | `firmware/kws_nano33/fe_tables.h`, `tools/fe_tables.json` |
| `convert_two_output.py` | `firmware/kws_nano33/model/with_aug_two_output_int8.tflite` (softmax + GAP outputs, same calibration as the published model) and `report/results/v0_two_output_equality.json` — needs TensorFlow, run with the model repo's environment |
| `export_model.py` | `firmware/kws_nano33/model_data.{h,cpp}`, `firmware/kws_nano33/model/model_meta.json` |
| `export_golden.py` | `tools/golden/golden_vectors.npz` + `golden_index.json` (50 test clips), `firmware/kws_nano33/golden_set.h` (3 audio + 5 int8 vectors) |

Run order: `convert_two_output.py` → `export_tables.py` → `export_model.py` → `export_golden.py --with-tflite`.
Tests: `python -m pytest tools/tests -q` (26 tests, 5 of them need `build_frontend_pc.py` first; `test_frontend_ref.py` needs the corpus checkout next to this repository).

## Implemented (M0, step 3 — 2026-11-01)

| Script / module | Output |
|---|---|
| `stream_scoring.py` | decision stage (K-mean, threshold, global suppression, energy gate) and Warden metrics — NumPy port of `src/streaming_eval.py`, 8 tests |
| `simulate_operating_point.py` | device pipeline over the saved isolated-clip stream (`../model-evaluation/results/stream/warden_stream_v2.wav`), grid of decision parameters → `report/results/operating_point_sim.{csv,json,md}`; per-hop probabilities cached in `tools/out/opsim/` |
| `noise_floor_check.py` | accuracy with a microphone noise floor replacing the digital-silence padding, with/without the frame-gate mitigation (`--gate-margin-db 6`) → `report/results/noise_floor_check.{json,md}` |

Recommended operating point from the simulation: hop 300 ms, K 2, threshold 0.85 (0.4 FP/min, 52 % correct on the
isolated-clip stream) or 0.55 (1.1 FP/min, 62 %), suppression 1.5 s; the device cycle (window finishing + Invoke) must
therefore stay within 300 ms.

## Implemented (M1 — board bring-up, 2026-11-01…03)

| Script | Purpose |
|---|---|
| `serial_log.ps1` | PowerShell logger for the board's USB serial port (.NET `SerialPort`, no pyserial): fixed duration, host timestamps, scheduled commands (`-Schedule "1=G 20|16=S"`), `-Quiet` for background runs |
| `pdm_analyze.py` | parses `pdm_rms` logs: noise floor per gain segment (p5 / median / min of the DC-free RMS, max peak), stream continuity (samples vs 16 kHz × time, short reads, stalls) → markdown table + JSON in `report/results/m1_pdm_*.json` |
| `ram_baseline_parse.py` | parses `ram_baseline` logs (linker map, heap and thread statistics, malloc probes) for the empty build and the build with the KWS buffers → side-by-side table + `report/results/m1_ram_baseline.json` with the derived static budget |

Logs go to `tools/out/pdm/` and `tools/out/ram/` (untracked); the summaries are in `report/results/m1_bringup.md`.

## Implemented (M2 — LiteRT Micro on the board, 2026-11-03)

| Script | Purpose |
|---|---|
| `m2_bench_parse.py` | parses `firmware/kws_nano33` logs: benchmark, cycles per operator group, V1 against LiteRT 2.3.0 and tf.lite 2.17 → table + `report/results/m2_bench.json` |
| `fc_scale_check.py` | what the library's single-scale dense layer costs: device pipeline on the PC, float head with correct and with single scale, 1,204 clips → `report/results/m2_fc_scale_check.json` |

## LiteRT for Microcontrollers library (M2, 2026-11-03)

| Script | Purpose |
|---|---|
| `generate_litert_micro_tree.sh` | Git Bash on Windows: clones `tensorflow/tflite-micro` at the pinned commit and runs the upstream `create_tflm_tree.py` for Cortex-M4 (softfp) with CMSIS-NN |
| `build_litert_micro_arduino.py` | turns that tree into the Arduino library `LiteRT_Micro`: file names, include paths, build defines, inline DSP intrinsics for GCC 7.2, Arduino port, provenance file |

## Implemented (M3 — frontend, 2026-11-03)

| Script | Purpose |
|---|---|
| `build_frontend_pc.py` | compiles `firmware/kws_nano33/frontend.c` into `tools/out/frontend_pc/frontend.dll` with the MSVC compiler of Visual Studio |
| `frontend_pc.py` | ctypes wrapper of that DLL |
| `frontend_pc_eval.py` | test V3: C frontend + int8 model on the test set against the NumPy and librosa paths → `report/results/v3_frontend_pc.json` |

`m3_board_compare.py` parses the frontend test of the board (`KWS_TEST_MODE` 2) and applies the V2 criteria → `report/results/m3_board_frontend.json`.

`serial_harness.py` drives the board in `KWS_TEST_MODE` 3: `--ping`, `--v1`, `--v2`, `--v4` → `report/results/device_v*.json`. V4 keeps every clip in `tools/out/m3/v4_clips.jsonl` and continues from there when restarted.
