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
