# report/results — verification, simulation and board bring-up outputs

| File | Produced by | Content |
|---|---|---|
| `v0_two_output_equality.json` | `tools/convert_two_output.py` | V0: two-output INT8 model vs the published model on the 1,204 test clips (softmax identical, GAP identical to the published MEAN tensor), quantization parameters, model sha256, submodule commit |
| `operating_point_sim.{csv,json,md}` | `tools/simulate_operating_point.py` | device-pipeline run over the saved isolated-clip stream (`warden_stream_v2.wav`, 602 s, 153 events): every decision configuration (hop × K × threshold × suppression × gate) with Warden metrics; recommended operating points; per-hop raw probabilities are cached in `tools/out/opsim/` (not tracked) |
| `noise_floor_check.{json,md}` | `tools/noise_floor_check.py` | accuracy of the device pipeline when white/pink noise at −60/−50/−40 dBFS replaces the digital-silence padding of the test clips, with and without the frame-gate mitigation |
| `m1_bringup.md` | M1 steps, written by hand from the arduino-cli / serial output | board bring-up: toolchain versions, first flash of `firmware/bringup/blink_hello` (flash/RAM baseline of the bare Mbed core, upload time, serial heartbeat check), PDM microphone results (noise floor vs gain, speech level, stability) |
| `m1_pdm_*.json` | `tools/pdm_analyze.py` on `tools/out/pdm/*.log` | per-gain noise floor (p5/median/min of DC-free RMS, max peak), speech level at +20 dB, far-from-laptop check, 10-minute continuity |

Device-side results (V1–V7, measurements) will be added here by `tools/serial_harness.py` and `tools/collect_measurements.py`.
