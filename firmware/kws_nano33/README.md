# firmware/kws_nano33 — Arduino sketch (LiteRT for Microcontrollers)

Target: Arduino Nano 33 BLE Sense Lite (nRF52840), core "Arduino Mbed OS Nano Boards" `arduino:mbed_nano` 4.6.0
(GCC 7-2017q4), FQBN `arduino:mbed_nano:nano33ble`; built with `arduino-cli` 1.5.1 (Arduino IDE 2.3.10 also works).
Development PC port: COM4 (board USB VID:PID 2341:805A).

Implemented so far: `frontend.{h,c}`, `kws_model.{h,cpp}`, `config.h`, board tests `selftest_model.cpp`, `selftest_frontend.cpp` and the serial test server `serial_proto.cpp` selected by `KWS_TEST_MODE`, and `kws_nano33.ino` as a start-up test only (V1 on 5 golden vectors, 20-run benchmark, per-operator
profile, no audio). Planned modules: `kws_nano33.ino` (state machine), `config.h`, `audio_capture.{h,cpp}` (PDM → 1-s ring buffer),
`frontend.{h,c}` (MFCC+Δ+ΔΔ+CMVN in plain C, shared with the PC tests), `fe_tables.h` and `model_data.{h,cpp}`
(generated), `kws_model.{h,cpp}` (LiteRT Micro wrapper), `decision.{h,cpp}`, `serial_proto.{h,cpp}`, `vad.{h,cpp}`,
and for stage 2 `head.{h,cpp}` + `head_init_data.h`.

## LiteRT for Microcontrollers

There is no current Arduino packaging. The official `Arduino_TensorFlowLite` is an archived December 2023 snapshot that mishandles the dense layer of models from recent converters
(`report/results/m2_litert_micro.md`), so the library is generated from `tensorflow/tflite-micro` commit `85c6999a1`. Prerequisites: Git for Windows, Python 3, the Arduino core, and GNU make:

```bash
winget install --id ezwinports.make --exact
```

Generate the sources, in Git Bash from the repository root (≈1 min, writes to `C:\lb`):

```bash
tools/generate_litert_micro_tree.sh
```

Turn them into the library `Documents/Arduino/libraries/LiteRT_Micro`:

```bash
python tools/build_litert_micro_arduino.py --tree C:/lb/tree --commit 85c6999a1f0032f64f3b0002431a357b451275c4
```

Build and flash. The first build compiles the whole library (≈9 min). Use the default build cache and `--output-dir`, not a long `--build-path`: object paths over 260 characters fail on Windows with `opening dependency file`.

```bash
arduino-cli compile -b arduino:mbed_nano:nano33ble --build-property "build.extra_flags=-O3" --build-property "compiler.cpp.extra_flags=-std=gnu++17 -include chrono" --output-dir build/kws_nano33 firmware/kws_nano33
```

```bash
arduino-cli upload -b arduino:mbed_nano:nano33ble -p COM4 --input-dir build/kws_nano33 firmware/kws_nano33
```

In Arduino IDE the same flags go into `platform.local.txt` next to the core's `platform.txt` (this route is not tested yet):

```
build.extra_flags=-O3
compiler.cpp.extra_flags=-std=gnu++17 -include chrono
```
