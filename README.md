# kws-ua-nano33ble — Ukrainian keyword spotting on Arduino Nano 33 BLE Sense

Real-time recognition of 13 Ukrainian keywords (так, ні, один, два, три, далі, тут, там, добре, ось, зараз, потім,
сьогодні) plus `_silence` / `_unknown_` on an **Arduino Nano 33 BLE Sense** (nRF52840, Cortex-M4F 64 MHz, 256 KB RAM,
1 MB flash, PDM microphone) with **LiteRT for Microcontrollers**. The DS-CNN model (30,415 parameters, INT8) comes from
[kws-model-evaluation](https://github.com/TarasBiletskyi/kws-model-evaluation) and was trained on the automatically built
corpus [common-voice-ua-kws-corpus](https://github.com/TarasBiletskyi/common-voice-ua-kws-corpus).

*Українською:* розпізнавання 13 українських ключових слів у реальному часі на мікроконтролері; вся обробка — від
мікрофона до рішення — виконується на платі. Детальний звіт буде у `report/`.

**Status:** milestone M0 (PC-side preparation) complete on 2026-11-01: constant tables, two-output INT8 model (V0 bit-exact), model C array, golden vectors, operating-point simulation (hop 300 ms / K 2 recommended) and microphone-noise-floor check (frame-gate mitigation). M1 step 1 (toolchain) done on 2026-11-01: arduino-cli 1.5.1, Arduino IDE 2.3.10 and the core `arduino:mbed_nano` 4.6.0 are installed and the board enumerates as `arduino:mbed_nano:nano33ble` (COM4 on the development PC). M1 step 2 done the same day: the first sketch (`firmware/bringup/blink_hello`) compiles (85.7 KB flash, 44.6 KB static RAM), flashes over COM4 in 9 s and streams a 1 Hz heartbeat over USB serial (see `report/results/m1_bringup.md`). M1 step 3 done on 2026-11-02 (`firmware/bringup/pdm_rms`): the PDM microphone ran 10 minutes at 16 kHz without a lost buffer; noise floor −52 dBFS and speech ≈ −28 dBFS at the chosen gain +20 dB (register 80); a 2 s warm-up after `PDM.begin()` and the frame gate are confirmed requirements. M1 step 4 done on 2026-11-03 (`firmware/bringup/ram_baseline`): the Mbed core takes 44.6 KB of static RAM, 32.8 KB of which is the main-thread stack our code runs on, and < 1 KB of heap at run time; the plan's full KWS budget (160 KB tensor arena + 32 KB audio ring + 15.5 KB log-mel ring, 211 KB as statics) boots next to it with 4.7 KB of heap region left, so the full model stays (firmware rule: heap region ≥ 8 KB, static ≤ 252 KB). **M1 complete.** M2 on 2026-11-03: the model runs on the board on LiteRT for Microcontrollers generated from current `tensorflow/tflite-micro` (`85c6999a1`, CMSIS-NN): arena 133,688 B, 291 KB flash, softmax bit-exact with tf.lite 2.17 on the 5 on-board golden vectors. One `Invoke` takes **1.97 s at `-O3`**, about 6.6× over the 300 ms budget and spread over all convolution layers, and accepted by the user's gate decision: the project continues with the published model, and retraining on a smaller input is deferred (see `report/results/m2_litert_micro.md`). M3 step 1 done the same day: `firmware/kws_nano33/frontend.c`, compiled for the PC, reaches the same 95.43 % as the training pipeline on the 1,204 test clips (V3); M3 step 2: on the board its int8 inputs are identical to the training pipeline on the 3 built-in golden clips; 1.96 ms per 10 ms frame, 47 ms per window. M3 complete: over USB serial the board reaches **95.35 %** on the whole test set (V4), 0.08 points from the PC; its softmax is bit-exact with tf.lite 2.17 on 50 golden clips of all 15 classes (V1), and the frontend matches on audio (V2). Next: M4 — microphone, decisions, live test.

## Layout

| Path | Contents |
|---|---|
| `firmware/kws_nano33/` | Arduino sketch: PDM capture, MFCC frontend in C, LiteRT Micro inference, decision stage, serial protocol (stage 2: on-device personalization) |
| `tools/` | Python tools (LiteRT runtime, no TensorFlow needed): constant export, golden vectors, serial test harness, operating-point simulation, measurement collection |
| `report/` | Short report (architecture, verification, measurement table), figures |
| `external/kws-model-evaluation/` | Git submodule pinned to the model repository commit used for the article (features, model, results) |

## Software platform

- Device: LiteRT for Microcontrollers (C++ runtime from `tensorflow/tflite-micro`), Arduino IDE 2.x / `arduino-cli`,
  "Arduino Mbed OS Nano Boards" core.
- PC: LiteRT Python runtime `ai-edge-litert` for all inference and verification; model conversion with the TensorFlow
  LiteRT converter (`tf.lite.TFLiteConverter`) in the model repository's environment.

### Toolchain versions used in this project (installed 2026-11-01, Windows 11)

| Component | Version / source |
|---|---|
| `arduino-cli` | 1.5.1 (winget `ArduinoSA.CLI`, MSI → `C:\Program Files\Arduino CLI`) |
| Arduino IDE | 2.3.10 (winget `ArduinoSA.IDE.stable`, per-user install) |
| Core "Arduino Mbed OS Nano Boards" `arduino:mbed_nano` | 4.6.0 (ArduinoCore-mbed 4.6.0), data dir `%LOCALAPPDATA%\Arduino15` (≈720 MB with tools) |
| Compiler (bundled with the core) | `arm-none-eabi-gcc` 7-2017q4 |
| Upload tools (bundled) | bossac 1.9.1-arduino2, openocd 0.11.0-arduino2, dfu-util 0.10.0-arduino1 |
| Board | Arduino Nano 33 BLE Sense **Lite** (Tiny Machine Learning Kit; u-blox NINA-B306 = nRF52840), FQBN `arduino:mbed_nano:nano33ble` (the Sense variants use the plain Nano 33 BLE entry), USB VID:PID 2341:805A in sketch mode |
| LiteRT for Microcontrollers (on the device) | Arduino library `LiteRT_Micro` generated from `tensorflow/tflite-micro` commit `85c6999a1` with CMSIS-NN kernels (see `firmware/kws_nano33/README.md`) |
| PC runtime | `ai-edge-litert` 2.3.0, Python 3.11 (`tools/requirements.txt`) |

## Build and run

Prerequisites on the PC (one-off):

```powershell
winget install --id ArduinoSA.CLI --exact          # optional: winget install --id ArduinoSA.IDE.stable --exact
arduino-cli core update-index
arduino-cli core install arduino:mbed_nano@4.6.0
arduino-cli board list                             # expect: COM<n>  Arduino Nano 33 BLE  arduino:mbed_nano:nano33ble
```

Compiling and flashing the sketch is documented from milestone M2 on; see `tools/README.md` for the PC side.

## License

MIT (code). The model and dataset keep the licenses of their own repositories.
