# kws-ua-nano33ble — Ukrainian keyword spotting on Arduino Nano 33 BLE Sense

Real-time recognition of 13 Ukrainian keywords (так, ні, один, два, три, далі, тут, там, добре, ось, зараз, потім,
сьогодні) plus `_silence` / `_unknown_` on an **Arduino Nano 33 BLE Sense** (nRF52840, Cortex-M4F 64 MHz, 256 KB RAM,
1 MB flash, PDM microphone) with **LiteRT for Microcontrollers**. The DS-CNN model (30,415 parameters, INT8) comes from
[kws-model-evaluation](https://github.com/TarasBiletskyi/kws-model-evaluation) and was trained on the automatically built
corpus [common-voice-ua-kws-corpus](https://github.com/TarasBiletskyi/common-voice-ua-kws-corpus).

*Українською:* розпізнавання 13 українських ключових слів у реальному часі на мікроконтролері; вся обробка — від
мікрофона до рішення — виконується на платі. Детальний звіт буде у `report/`.

**Status:** work in progress — milestone M0 (PC-side preparation). Nothing runs on the board yet.

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

## Build and run

To be written at milestone M1/M2 (board bring-up, model on the board). Until then see `tools/README.md` for the PC side.

## License

MIT (code). The model and dataset keep the licenses of their own repositories.
