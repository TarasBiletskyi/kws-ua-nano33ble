# firmware/kws_nano33 — Arduino sketch (LiteRT for Microcontrollers)

Target: Arduino Nano 33 BLE Sense Lite (nRF52840), core "Arduino Mbed OS Nano Boards" `arduino:mbed_nano` 4.6.0
(GCC 7-2017q4), FQBN `arduino:mbed_nano:nano33ble`; built with `arduino-cli` 1.5.1 (Arduino IDE 2.3.10 also works).
Development PC port: COM4 (board USB VID:PID 2341:805A).

Planned modules: `kws_nano33.ino` (state machine), `config.h`, `audio_capture.{h,cpp}` (PDM → 1-s ring buffer),
`frontend.{h,c}` (MFCC+Δ+ΔΔ+CMVN in plain C, shared with the PC tests), `fe_tables.h` and `model_data.{h,cpp}`
(generated), `kws_model.{h,cpp}` (LiteRT Micro wrapper), `decision.{h,cpp}`, `serial_proto.{h,cpp}`, `vad.{h,cpp}`,
and for stage 2 `head.{h,cpp}` + `head_init_data.h`.

LiteRT Micro library: plan A — the official Arduino library `Arduino_TensorFlowLite` (manual install from
`tensorflow/tflite-micro-arduino-examples`); plan B — a library generated from upstream `tensorflow/tflite-micro`
with CMSIS-NN kernels. Build instructions follow at milestone M2.
