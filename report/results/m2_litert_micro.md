# M2 — LiteRT for Microcontrollers on the board

## Step 1 (2026-11-03) — library installed, compile check with the real model

Nothing was flashed in this step. The check sketch only proves that the library links with the model for this board.

### 1a — the library (plan A)

| Item | Value |
|---|---|
| Source | official `tensorflow/tflite-micro-arduino-examples`, cloned into `Documents/Arduino/libraries/Arduino_TensorFlowLite` |
| Commit | `7948b85e020470a30a83bf00e10fc43b486bfcba` (2023-12-21), the repository's last commit |
| `library.properties` | `Arduino_TensorFlowLite` 2.4.0-ALPHA |
| Size on disk | 44 MB |
| Arduino library index | not listed under this name, only third-party snapshots, which the plan excludes, so the install is a manual clone |
| Declared dependencies | `ArduinoBLE`, `Arduino_LSM9DS1`: not installed and not needed (they serve the bundled examples; the build passes without them) |

`library.properties` declares `precompiled=full`, but the clone holds no `.a` file, so the builder reports "not found" and
compiles all sources. Those two lines in the build log are expected.

### 1b — kernels

The library contains **only the CMSIS-NN versions** of `conv`, `depthwise_conv`, `fully_connected`, `softmax`, `pooling`,
`add`, `mul` and `svdf`: the reference `.cpp` files are absent, so there is no build switch between the two and no
"without CMSIS-NN" comparison is possible with this library. `MEAN` runs on the portable `reduce.cpp`.

Instruction counts from the linked ELF (`arm-none-eabi-objdump`, by address range; DSP = SMLAD, SXTB16, SXTAB16, PKHBT and similar):

| Kernel | Used for (per the CMSIS-NN wrappers' conditions) | MACs per window | Instructions | DSP |
|---|---|---|---|---|
| `arm_nn_mat_mult_nt_t_s8` | 5 × 1×1 convolutions (via `arm_convolve_1x1_s8_fast`) | 20 M | 662 | **110** |
| `arm_nn_mat_mult_kernel_s8_s16` | first convolution 10×4 (via `arm_convolve_s8`) | 2.5 M | 481 | 30 |
| `arm_depthwise_conv_3x3_s8` | 5 × depthwise 3×3 | 2.8 M | 520 | **0** |
| `arm_nn_softmax_common_s8` | softmax | — | 1,251 | 0 |

So the DSP path is really compiled in (`ARM_MATH_DSP` follows from `__ARM_FEATURE_DSP` with `-mcpu=cortex-m4`), and ≈80 %
of the MACs run in the most heavily DSP-optimised kernel. Which kernel each layer reaches is read from the wrapper source
and still has to be confirmed by the per-operator profile at step 4 of this milestone.

**Latency lever to test at step 4:** the core compiles everything with `-Os` (`variants/ARDUINO_NANO33BLE/cflags.txt`,
`cxxflags.txt`). A later `-O2`/`-O3` in `build.extra_flags` overrides it.

### 1c — build with the two-output model and a 160 KB arena

The sketch: `model_data.cpp` (57,960 B, sha256 `394c9fad…`), `MicroMutableOpResolver<5>` (`AddConv2D`, `AddDepthwiseConv2D`,
`AddMean`, `AddFullyConnected`, `AddSoftmax`; these are all the operators of the model), `MicroInterpreter`,
`AllocateTensors()`, `arena_used_bytes()`.

| Item | Result |
|---|---|
| Compile | OK, 0 warnings, 297 s for the first build of the whole library |
| Flash | **228,232 B** (23 % of 983,040) |
| Static RAM | **210,344 B** (80 %), of which the arena is 163,840 B |
| Flash added by the library | ≈82 KB (228,232 − 88,272 of the empty sketch − 57,960 of the model) |
| Static RAM added by the library | ≈2 KB (210,344 − 163,840 − 44,552 of the empty sketch) |

**Budget consequence.** With the rule from M1 step 4 (static ≤ 252,416 B, so that the heap region stays ≥ 8 KB) and both rings in
place (47,520 B), the arena may take up to **≈158,400 B (≈154 KB)**: 252,416 − 46,504 − 47,520. The expected 135–145 KB fits; the
real figure comes from `arena_used_bytes()` at step 2.

### 1d — pitfall: Windows path length

The first build failed with `fatal error: opening dependency file …\log_lut.c.d: No such file or directory`. The cause was a long
`--build-path`: together with the library's deep directories the object path passed 260 characters, the limit for GCC 7 on
Windows. **Do not pass a long `--build-path` for this library.** The default cache of `arduino-cli` and Arduino IDE
(`%LOCALAPPDATA%\arduino\sketches\<hash>`) is short enough (188 characters for that file), and `--output-dir` is safe because it only
copies the results.

### Not verified yet

No code from this library has run on the board. In particular, the library dates from 2023 and the model was converted with
TensorFlow 2.17, so whether `AllocateTensors()` and `Invoke()` accept the model is the first thing step 2 checks; plan B (a library
generated from current `tflite-micro`) stays open until then.

## Step 2 (2026-11-03) — the model on the board: `firmware/kws_nano33` (first version)

The first version of the real firmware: `kws_model.{h,cpp}` (the wrapper of the plan; all runtime calls live there), `config.h`,
`kws_nano33.ino` (start-up test). One flash. On start it builds the interpreter, runs **one** `Invoke` on golden vector 0 and
repeats a report every 5 s. Log `tools/out/m2/step2_first_invoke.log` (untracked).

| Item | Result |
|---|---|
| Runtime accepts the model (2023 library, converted with TF 2.17) | **yes**: schema 3, `AllocateTensors` and `Invoke` return OK, plan B not needed for this |
| Build | flash 236,216 B (24 %), static 210,400 B (80 %) with a 160 KB arena; 492 s for the first build of this sketch |
| Upload | 15 s, 58 × 4 KB pages |
| **Arena used (`arena_used_bytes()`)** | **134,920 B**, the low end of the 135–145 KB estimate |
| `AllocateTensors` | 27 ms |
| Main-thread stack used | 1,336 B of 32,768 B (CMSIS-NN scratch sits in the arena), so ≈31 KB stay free for the frontend |
| Heap in use | 3,404 B (876 B at M1 step 4; the source of the extra 2.5 KB is not identified; no effect within this budget) |
| CPU clock, instruction cache | 64 MHz, cache on |
| Softmax vs the PC (LiteRT 2.3.0), golden vector 0 | **identical** in all 15 values, top class 0 = label |
| GAP vs the PC | differs by ±1 in 14 of 64 values, as expected from different rounding in the kernels; only the softmax drives stage 1, stage 2 has to tolerate this |
| **`Invoke`, one single run** | **2,400,587 µs = 2.40 s** |

### Memory: settled

Static without the arena 46,560 B + arena 134,920 + 2,048 B margin + both rings 47,520 B = 231,048 B, which leaves **≈21 KB** under the
252,416 B limit from M1 step 4. That also covers stage 2 (≈12 KB).

### Latency: the plan's estimate was wrong by about a factor of 5

| Quantity | Value |
|---|---|
| Cycles of one `Invoke` | 153.6 M (2.40 s × 64 MHz) |
| Cycles per MAC | ≈6.0 (25.4 M MACs) |
| Budget at hop 300 ms | 19.2 M cycles, so **≈8× too slow** |

The plan expected 0.3–0.6 s, which assumed about 1 cycle per MAC. Only the MACs were counted; requantisation of every output
value, the depthwise kernel without DSP instructions (see 1b), the weights read from flash and `-Os` were not. This is one sample;
the split per operator is still unmeasured.

Consequences, none of which is decided yet:

- **Plan rule "Invoke > 0.8 s → model variant"** is triggered unless a compiler or memory change brings it below that.
- **Streaming is impossible as designed.** `Invoke` blocks for longer than the 1-s audio ring holds, so audio would be lost and no
  word could be scored on a 300 ms hop.
- **Rough scale of the variants, assuming the same cycles per MAC (an extrapolation, not a measurement):**

| Variant | MACs | `Invoke` at 6 cycles/MAC |
|---|---|---|
| Published model | 25.4 M | 2.40 s (measured) |
| First convolution stride (4,2) | ≈13 M | ≈1.2 s |
| + 32 channels | ≈7 M | ≈0.66 s |
| What a 300 ms cycle allows | ≈3 M | 0.3 s |

The stride variant from the plan alone does not reach the budget at the current speed.

## Steps 3–4 (2026-11-03) — V1, benchmark, per-operator profile, `-O3`

`firmware/kws_nano33` runs 25 `Invoke` calls on start: 5 for V1 on the 5 on-board golden vectors, then 20 for the benchmark, with a
cycle profiler (DWT counter behind the library's `MicroProfilerInterface`) and the instruction-cache counters. Two flashes, the second
with `--build-property "build.extra_flags=-O3"`, which reaches the library and the sketch after the core's `-Os` (confirmed in the
verbose build log and by `opt=speed` in the `HELLO` line). `-O2` was not tested. Logs `tools/out/m2/bench_{Os,O3}.log` (untracked),
JSON `m2_bench.json` (`tools/m2_bench_parse.py`).

### Latency

| | `-Os` (core default) | `-O3` |
|---|---|---|
| `Invoke` min / mean / max, 20 runs | 2.386 / **2.387** / 2.387 s | 1.968 / **1.969** / 1.970 s |
| Cycles, mean | 152.8 M | 126.0 M |
| Cycles per MAC | 6.01 | 4.96 |
| Over the 300 ms budget | ×8.0 | **×6.6** |
| Flash | 253,336 B | 274,512 B |
| Instruction-cache hits | 99.3 % | 99.1 % |
| Arena | 134,920 B | 134,920 B |
| Main stack used | 1,368 B | 1,160 B |

Runs repeat to within ±0.1 % and do not depend on the input. `-O3` gives −17.5 %, and its build needs that property, or a
`platform.local.txt` in Arduino IDE.

### Where the time goes

| Operators | MACs | `-Os`, ms | share | `-O3`, ms | share | cycles/MAC at `-O3` |
|---|---|---|---|---|---|---|
| 5 × `CONV_2D` 1×1 | 20 M | 1,096 | 45.9 % | 968 | 49.1 % | 3.1 |
| 5 × `DEPTHWISE_CONV_2D` 3×3 | 2.8 M | 677 | 28.4 % | 586 | 29.8 % | 13.4 |
| first `CONV_2D` 10×4 | 2.5 M | 320 | 13.4 % | 226 | 11.5 % | 5.8 |
| `MEAN` (no CMSIS-NN kernel) | — | 294 | 12.3 % | 190 | 9.6 % | — |
| `FULLY_CONNECTED` + `SOFTMAX` | — | 0.2 | 0 % | 0.3 | 0 % | — |

No single operator is the culprit: replacing `MEAN` with a fast loop would save at most ≈190 ms. The time is spread over the
10 layers that carry 49×20×64 values, so only a smaller model or fewer invocations can close a gap of ×6.6.

**Plan rule "`Invoke` > 0.8 s → model variant": triggered, and compiler options do not change that.**

### V1: top class 5/5, but the softmax is wrong — library defect

| Vector (label) | top class device / PC | softmax max\|Δ\| vs LiteRT 2.3.0 | vs tf.lite 2.17 | GAP max\|Δ\| (values differing of 64) |
|---|---|---|---|---|
| 0 (далі) | 0 / 0 | 0 | 0 | 1 (14) |
| 1 (далі) | 0 / 0 | **12** | 12 | 1 (13) |
| 2 (далі) | 0 / 0 | **34** | 30 | 1 (9) |
| 3 (два) | 1 / 1 | 1 | 1 | 1 (13) |
| 4 (два) | 1 / 1 | 0 | 1 | 1 (19) |

The results are the same at `-Os` and `-O3`. The criterion "argmax identical" holds; bit-exact match with one PC runtime does not. It
is also weaker than the plan's V1 (5 vectors of 2 classes instead of 50 of 15), so V1 is not closed.

**Cause.** The TF 2.17 converter quantizes the dense layer **per class** (15 weight scales, 0.80–1.11 of the first). The 2023 library takes
`scale()->Get(0)` in `micro_allocator.cpp` and builds one multiplier in `fully_connected_common.cpp`, without an error. So on the board logit
*i* is multiplied by s₀/sᵢ. A simulation of exactly this reproduces the board and the correct scales do not:

| Vector | board softmax, top 2 of 256 | simulated single scale | correct scales |
|---|---|---|---|
| 1 | 236, 20 | 236.0, 20.0 | 247.6, 8.4 |
| 2 | 191, 64 | 188.0, 66.9 | 220.2, 34.6 |

An explanation through the ±1 in GAP was tested first and rejected: applied to the board's GAP, the correct head gives 220/35, not 191/64.

**Cost on the whole test set** (`tools/fc_scale_check.py`, PC simulation with a float head, 1,204 clips; JSON `m2_fc_scale_check.json`):

| | correct scales | single scale (board) |
|---|---|---|
| Accuracy | 95.51 % | **94.35 %** (17 clips change class) |
| Correct with confidence ≥ 0.85 | 88.29 % | 87.13 % |
| Recall of «так» (largest class, scale 1.11) | 85.5 % | 81.0 % |

Fix options, none applied yet: **(A)** reconvert with the single-scale switch for dense layers that TF 2.17 has
(`_experimental_disable_per_channel_quantization_for_dense_layers`), which leaves only the dense layer's quantization different from the published
model and needs a new accuracy check; **(B)** a library generated from upstream `tflite-micro`, whose `fully_connected_common.cc` handles
per-channel weights (toolchain effort on Windows and GCC 7 unknown); **(C)** compute the head from the GAP vector in the firmware.

### Options for the gate (the user's decision)

| Path | What changes | `Invoke` | Open |
|---|---|---|---|
| **1. Event-triggered, published model** | Instead of a fixed hop, run `Invoke` once when an energy detector finds a word in the ring | 1.97 s measured, the board is deaf to new decisions for ≈2 s after each word | isolated words with pauses only, as the article recommends; accuracy to be measured live |
| **2. Smaller model, streaming** | Retrain in `kws-model-evaluation`, hop 300 ms stays | see below | accuracy unknown until trained; no longer the published model |

Rough scaling of the `-O3` profile by feature-map size, channels and blocks. **These are estimates, not measurements**, and they exclude the
frontend's window finish, which is not measured yet:

| Variant | Estimated `Invoke` |
|---|---|
| Published | 1.97 s (measured) |
| stride (4,2) | ≈1.0 s |
| stride (4,2), 32 channels | ≈0.43 s |
| stride (4,2), 32 channels, 3 blocks | ≈0.30 s, no margin |

The stride variant named in the plan is far from enough, and this replaces the extrapolation of step 2.

### The defect is a known upstream one (checked 2026-11-03)

The installed library is old, not new: `tensorflow/tflite-micro-arduino-examples` is archived, last pushed on 2023-12-21. The converter later switched dense
layers to per-channel scales, and upstream `tflite-micro` dealt with the consequences after that snapshot:

| Date | Upstream change | In the installed library |
|---|---|---|
| 2024-05-29 | [#2602](https://github.com/tensorflow/tflite-micro/pull/2602): a runtime error instead of silently invalid output; names the converter switch of fix (A) as the workaround | no, so the failure is silent here |
| 2024-09-27 | [#2673](https://github.com/tensorflow/tflite-micro/pull/2673): per-channel support in the reference and CMSIS-NN kernels (the basis of fix (B)) | no |

So fix (A) is the workaround the runtime's authors recommend.

### Library choice under review (2026-11-03)

The task requires LiteRT for Microcontrollers. `Arduino_TensorFlowLite` 7948b85 is named by Google's documentation, but it is an archived December 2023 snapshot, so all M2 numbers above
describe that old runtime. The maintained code is `tensorflow/tflite-micro` (`main`, active, no release tags), which has no current Arduino packaging: the documented regeneration script
`test_arduino.sh` no longer exists there. The way forward is a library generated from a pinned commit with `create_tflm_tree.py`. Known requirements: `make` (absent on the development PC, WSL not installed) and C++17
(the core builds with `gnu++14` on GCC 7.2; whether the code compiles with it is untested). Fix (A) above is not needed if this works. Flash, arena, latency and V1 must then be measured again.

## Step 5 (2026-11-03) — library replaced: LiteRT for Microcontrollers from current upstream

The firmware now runs on `LiteRT_Micro`, a library generated from `tensorflow/tflite-micro` commit `85c6999a1` with CMSIS-NN kernels
(`tools/generate_litert_micro_tree.sh`, `tools/build_litert_micro_arduino.py`). The script was checked on an empty directory: identical tree,
609 files. `Arduino_TensorFlowLite` is no longer used. The sections above measure that old library and stay as history. The published model is unchanged, so
fix (A) is not needed.

| | old library, `-O3` | **LiteRT Micro, `-O3`** |
|---|---|---|
| `Invoke` min / mean / max, 20 runs | 1.968 / 1.969 / 1.970 s | 1.968 / **1.970** / 1.971 s |
| Over the 300 ms budget | ×6.6 | **×6.6** |
| Softmax vs tf.lite 2.17 reference, 5 vectors | up to 30/256 apart | **identical on 5/5** |
| Softmax vs LiteRT 2.3.0 | up to 34/256 apart | up to 4/256 apart, identical on 3/5 |
| GAP vs LiteRT 2.3.0 | ±1 | ±1 |
| Top class | 5/5 | 5/5 |
| Arena | 134,920 B | **133,688 B** |
| Static RAM with a 160 KB arena | 211,232 B | 209,608 B |
| Heap in use | 3,404 B | 876 B, equal to the empty sketch of M1 step 4 |
| Flash | 274,512 B | 291,272 B (29 %) |

Where the time goes, `-O3`:

| Operators | old library, ms | LiteRT Micro, ms | cycles/MAC |
|---|---|---|---|
| 5 × `CONV_2D` 1×1 | 968 | 1,010 | 3.2 |
| 5 × `DEPTHWISE_CONV_2D` 3×3 | 586 | 523 | 11.9 |
| first `CONV_2D` 10×4 | 226 | 207 | 5.3 |
| `MEAN` | 190 | 231 | — |

**V1.** The dense-layer defect is gone. The plan's criterion (bit-exact with at least one PC runtime) now holds for the softmax on these 5 vectors, which cover only 2 classes;
the plan's 50 vectors of 15 classes are still open. **Latency is unchanged,** so the gate decision stands with the same numbers. Memory: static without the arena 45,768 B + arena
133,688 + 2,048 B + rings 47,520 B = 229,024 B, ≈23 KB under the 252,416 B limit.

Build details that matter:

- **Compile flags:** `--build-property "build.extra_flags=-O3" --build-property "compiler.cpp.extra_flags=-std=gnu++17 -include chrono"`. C++17 is required by LiteRT Micro. GCC 7.2 handles it. `-include chrono` is needed because `Arduino.h` defines an `abs` macro before Mbed pulls in `<chrono>`, whose C++17 `abs` it breaks.
- **DSP intrinsics:** `__sxtb16`, `__smlabb`, `__smlatt` are absent from `arm_acle.h` of GCC 7.2, so the generated config header provides them inline. A first version made them ordinary functions: the board then took 2.46 s and the depthwise layers 1,005 ms instead of 523 ms. That `-Os` run gave 2.74 s; `-Os` was not measured again with the inline version.
- **Arduino IDE:** the flags go into `platform.local.txt` next to the core's `platform.txt`. This route has **not been tested yet**, and the task asks for the IDE.

## Why 2 seconds: comparison with published keyword spotting on Cortex-M (2026-11-03)

| | This model | MLPerf Tiny reference DS-CNN |
|---|---|---|
| Input | 98×39 = 3,822 values | 49×10 = 490 values |
| Blocks, filters | 5, 64 | 4, 64 |
| MACs | 25.4 M | 2.66 M (computed here from the layer sizes; the paper does not state it) |
| Clock | 64 MHz | 120 MHz |
| `Invoke` | 1.97 s, measured | 100.72 ms on the TFLM baseline, as listed by Plumerai (a vendor); official optimised results 72–75 ms |
| Cycles per MAC | 4.96 | ≈4.5 |
| Top-1 accuracy | 95.4 % (15 classes, Ukrainian) | 91.6 % quantized (12 classes, English) |

The board's speed per MAC is normal, so no measurement or configuration fault is indicated. Scaling the reference by MACs (×9.5) and clock (×1.9) predicts ≈1.8 s against the measured 1.97 s. The time comes from
the model's size: 7.8× larger feature maps than the reference (49×20 vs 25×5), caused by the 10 ms hop and the Δ/ΔΔ channels.

The same architecture on a 49×10 input would take 3.24 M MACs with 5 blocks: ≈0.25 s at the measured cycles per MAC, an extrapolation that excludes the frontend. That needs retraining and a frontend with a 40 ms window, 20 ms hop and 10 coefficients. Its accuracy on this corpus is unknown.
Sources: MLPerf Tiny (arXiv 2106.07597), `mlcommons/tiny` `keras_model.py`, Hello Edge (arXiv 1711.07128), Plumerai MLPerf Tiny 1.0 post.

## Gate decision (2026-11-03)

**The user's decision: continue with the published model and accept its latency.** The operating mode is settled at M4: a fixed hop is not possible with a blocking 1.97 s `Invoke`, so triggering `Invoke` on detected speech is the candidate.

**Deferred, to be resumed later:** retraining the same DS-CNN on a 49×10 input (40 ms window, 20 ms hop, 10 MFCC, no Δ/ΔΔ) for continuous listening. Estimate ≈0.25 s with 5 blocks or ≈0.21 s with 4, extrapolated and excluding the frontend; accuracy on this corpus unknown; it changes `kws-model-evaluation`, the tables, the model export and the frontend.

Still open from M2: V1 on the plan's 50 vectors of 15 classes (needs the serial harness of M3), the Arduino IDE build route, and setting `KWS_ARENA_BYTES` to `arena_used_bytes()` + 2 KB.
