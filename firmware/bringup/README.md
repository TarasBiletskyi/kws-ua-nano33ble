# firmware/bringup — board bring-up sketches (milestone M1)

Small sketches that each prove one thing. They are not part of the KWS firmware in `firmware/kws_nano33/`.

| Sketch | Step | Proves |
|---|---|---|
| `blink_hello/` | M1.2 | the toolchain flashes the board; LED pattern (3 × 80 ms flashes, 1 s period) and a 115200-baud heartbeat `HB n=<count> ms=<uptime>` over USB serial |
| `pdm_rms/` | M1.3 | PDM microphone at 16 kHz: one `RMS` line per 100 ms (DC-free RMS and peak in dBFS, sample/callback counts), `SUM` every 60 s, commands `G <0..80>` (gain register), `S`, `Z`; stall watchdog that restarts the driver. Used for the noise-floor sweep, the speech-level check and the 10-minute stability run |
| `ram_baseline/` | M1.4 | RAM baseline of the Mbed core: linker map (`.data`/`.bss`/heap/ISR stack), Mbed heap and per-thread stack statistics, malloc probes, printed every 2 s. Build B (`-DRAM_BASELINE_WITH_KWS_BUFFERS=1 -DKWS_ARENA_KB=160`) adds the plan's static KWS buffers (arena 160 KB, audio ring 32,000 B, log-mel ring 15,520 B) and proves that they boot next to the core |

Build and flash (take the port name from `arduino-cli board list`; run from the repository root):

```powershell
arduino-cli compile -b arduino:mbed_nano:nano33ble --output-dir build/blink_hello firmware/bringup/blink_hello
arduino-cli upload  -b arduino:mbed_nano:nano33ble -p COM4 --input-dir build/blink_hello firmware/bringup/blink_hello
arduino-cli compile -b arduino:mbed_nano:nano33ble --build-property "build.extra_flags=-DRAM_BASELINE_WITH_KWS_BUFFERS=1 -DKWS_ARENA_KB=160" --output-dir build/ram_baseline_b firmware/bringup/ram_baseline
```

Log the serial output and analyse it (PowerShell logger, no pyserial needed; Python with NumPy for the PDM analysis):

```powershell
.\tools\serial_log.ps1 -Port COM4 -Seconds 70 -Out tools\out\pdm\sweep.log -Quiet -Schedule "1=G 20|16=S|16.5=G 40|31.5=S|32=G 60|47=S|47.5=G 80|69=S"
python tools\pdm_analyze.py tools\out\pdm\sweep.log --settle 2 --json report\results\m1_pdm_sweep.json
.\tools\serial_log.ps1 -Port COM4 -Seconds 12 -Out tools\out\ram\ram_baseline_b.log -Quiet
python tools\ram_baseline_parse.py --log A=tools\out\ram\ram_baseline_a.log --log B=tools\out\ram\ram_baseline_b.log --json report\results\m1_ram_baseline.json
```

The same works from Arduino IDE 2.x: open the sketch folder, select "Arduino Nano 33 BLE" and the port, Upload.
Results are logged in `report/results/m1_bringup.md`.
