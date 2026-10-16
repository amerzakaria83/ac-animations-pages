# AWR1843 Radar Studio

A desktop app, a command-line tool and a Python library for controlling a **TI AWR1843BOOST**
mmWave radar over USB and getting its data out.

The app sends configurations to the radar, starts and stops it, and changes detection
settings while it runs. It shows everything the firmware sends in real time and records it
all to disk (CSV / JSON / NumPy). A built-in **simulator** lets you try the app without the
board.

## What you can get out of the radar

| Data | Units | Config to use |
|---|---|---|
| 3D point cloud: x, y, z, radial velocity (plus range, azimuth, elevation, computed here) | m, m/s, ° | any |
| SNR and noise level of each point | dB | any (`guiMonitor` detectedObjects = 1) |
| Range profile (zero-Doppler power vs. range) | dB (relative) | any |
| Noise profile | dB (relative) | `awr1843_heatmaps.cfg` |
| Range-Doppler heatmap | dB (relative) | `awr1843_heatmaps.cfg` |
| Range-azimuth heatmap (complex virtual-antenna samples; the app does the angle FFT) | – | `awr1843_heatmaps.cfg` |
| DSP/CPU statistics: processing time, UART transmit time, margins, CPU load | µs, % | any |
| Temperature of 10 on-chip sensors (RX0-3, TX0-2, PM, DIG0-1) | °C | any |
| Firmware/SDK version, frame counter, CPU cycle timestamp | – | any |

> **Raw ADC samples are not available over USB.** The out-of-box firmware only sends processed
> data on the UART. Raw I/Q capture needs the separate **DCA1000EVM** board and TI mmWave Studio.

## 1. Prepare the board (one time)

1. **Power:** the AWR1843BOOST needs its **5 V / 3 A barrel supply**. USB alone powers only
   the on-board XDS110 debugger, not the radar.
2. **Flash the out-of-box demo:** install the TI **mmWave SDK 3.x** (3.5 or 3.6) and
   **UniFlash**. Flash `xwr18xx_mmw_demo.bin` from
   `mmwave_sdk_03_0x_xx_xx/packages/ti/demo/xwr18xx/mmw/`.
   - Flashing mode: **SOP0 and SOP2 on**, then press reset (NRST), then flash.
   - Functional mode (normal use): **SOP2 on, SOP0 off**, then power-cycle or press NRST.
   - The switch positions are shown in the AWR1843BOOST user guide (SWRU546).
3. **Drivers:**
   - *Windows:* the XDS110 driver is installed with UniFlash or CCS. Device Manager should show
     **"XDS110 Class Application/User UART (COMx)"** (CLI) and
     **"XDS110 Class Auxiliary Data Port (COMy)"** (data).
   - *Linux:* the board appears as `/dev/ttyACM0` (CLI) and `/dev/ttyACM1` (data). Add yourself
     to the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and back in). If
     ModemManager is running, it may grab the ports; stop or uninstall it.
   - *macOS:* ports appear as `/dev/cu.usbmodem…1` (CLI) and `…4` (data).

## 2. Install and start

You need Python 3.9 or newer.

**Windows:** double-click `start_windows.bat`. **Linux / macOS:** run `./start.sh`.
Both create a local virtual environment on the first run and then open the app.

Or install by hand:

```bash
cd awr1843-radar
python -m pip install -r requirements.txt
python -m awr1843            # opens the GUI
python -m awr1843 gui --sim  # GUI with the simulator preselected (no hardware needed)
```

(`pip install .` also works and gives you an `awr1843` command.)

## 3. Using the app

The left panel is laid out in order of use:

1. **Connection:** click *Detect ports* (the XDS110 CLI and data ports are recognised
   automatically), then *Connect*. The app asks the firmware for its version to confirm that the
   demo is running. Choose *Simulator* to work without hardware.
2. **Configuration:** pick a bundled `.cfg` or load your own (for example, one exported from TI's
   mmWave Demo Visualizer). The app shows what the configuration means: range and velocity
   resolution, max range and velocity, frame rate, and **how much of the UART bandwidth it
   needs**. It warns you if the configuration cannot work.
3. **Sensor:** *Send config & Start* sends the configuration and starts the radar. *Stop* and
   *Resume* (`sensorStart 0`) pause and restart it without reconfiguring.
4. **Live tuning:** toggle static-clutter removal and change the CFAR detection thresholds while
   the radar runs. The app stops the radar, sends the commands and resumes it.
5. **Record / replay:** *Record* saves everything to `recordings/<timestamp>/`. *Replay…* plays
   a recording back through the same views. *Export…* converts a raw capture to CSV/NPZ.

Tabs:

- **Point cloud:** top view, side view, range-velocity scatter and the range profile with the
  detections marked. You can colour points by velocity, SNR or height, keep points from several
  frames on screen, filter by SNR and hide static points.
- **Heatmaps:** range-Doppler map and a range-azimuth map in x/y coordinates, with the detected
  points drawn on top. Use *Mirror azimuth* if left and right look swapped.
- **Telemetry:** points per frame, CPU load, timing margins and temperatures over time.
- **Points table:** every value of every point in the latest frame.
- **CLI console:** type any demo command, for example `version`, `queryDemoStatus`, `help` or
  `clutterRemoval -1 1`. The full CLI traffic is shown here.
- **Config editor:** edit the configuration before sending it. The text shown is exactly what
  gets sent.

### Bundled configurations (`awr1843/configs/`)

| File | Description |
|---|---|
| `awr1843_default.cfg` | 3D point cloud, 4.4 cm range resolution, ~10 m range, ±2 m/s, 10 fps |
| `awr1843_heatmaps.cfg` | The default plus noise profile and range-Doppler and range-azimuth heatmaps at 3.3 fps. The 921 600 baud UART can't carry heatmaps at 10 fps. |
| `awr1843_sdk_profile_3d.cfg` | TI's reference `profile_3d.cfg` from SDK 3.5, unchanged (±1 m/s) |

## 4. Command line

```bash
python -m awr1843 ports                        # list serial ports and detect the radar
python -m awr1843 info awr1843/configs/awr1843_heatmaps.cfg   # what a cfg means
python -m awr1843 send version queryDemoStatus # send raw CLI commands
python -m awr1843 run                          # configure + stream (status line)
python -m awr1843 run --points                 # print every detected point
python -m awr1843 run --record --duration 60   # record one minute to recordings/<timestamp>
python -m awr1843 run --cfg my.cfg --cli-port COM3 --data-port COM4
python -m awr1843 export recordings/<ts>/raw_data.bin   # raw capture -> CSV / NPZ
```

Add `--sim` to `run` or `send` to use the simulator.

## 5. Python API

```python
from awr1843 import RadarDevice
from awr1843.recorder import Recorder

with RadarDevice.connect() as radar:          # ports auto-detected (or pass "COM3", "COM4")
    print(radar.version())
    radar.configure(open("awr1843/configs/awr1843_default.cfg").read())
    print(radar.params.summary())             # resolution, max range, ...

    for frame in radar.frames(timeout=2):     # blocking iterator
        pts = frame.points                    # NumPy structured array
        print(frame.frame_number, len(pts), pts["x"], pts["y"], pts["velocity"], pts["snr_db"])
        frame.range_profile_db                # np.ndarray or None
        frame.range_doppler_db                # [range, doppler] or None
        frame.azimuth_static                  # complex [range, virtual antenna] or None
        frame.stats, frame.temperature        # dicts or None

    radar.apply_live(["clutterRemoval -1 1"]) # stop, change, resume
```

Use `RadarDevice.simulated()` in place of `RadarDevice.connect()` to run without the board.
`frame.points` has the fields `x y z velocity range azimuth_deg elevation_deg snr_db noise_db`.

## 6. Recording format

Each recording folder contains:

| File | Content |
|---|---|
| `raw_data.bin` | every byte from the data port; lossless and replayable |
| `config.cfg` | configuration that was used |
| `points.csv` | `frame, host_time, sub_frame, x, y, z, velocity, range, azimuth_deg, elevation_deg, snr_db, noise_db` |
| `telemetry.csv` | per frame: point count, CPU cycles, the 6 DSP/CPU stats, 10 temperatures |
| `frames.jsonl` | the same per-frame metadata as JSON lines |
| `range_profile.csv`, `noise_profile.csv` | one row per frame, one column per range bin (dB) |
| `heatmaps.npz` | `frame_numbers`, `range_doppler_db [F,R,D]`, `azimuth_static [F,R,A]` (complex) |

```python
import numpy as np, pandas as pd
pts = pd.read_csv("recordings/20250101_120000/points.csv")
hm = np.load("recordings/20250101_120000/heatmaps.npz")
```

## Coordinate system

These are the firmware's coordinates. The radar looks along **+y** (boresight), **x** is
lateral and **z** is height. A positive velocity means the target is moving away.

## Troubleshooting

| Symptom | Fix |
|---|---|
| *No response* on Connect | Board not powered by the 5 V supply, or still in flashing mode (SOP0 on); press NRST. Check that the demo firmware is flashed. Check you didn't swap the CLI and data ports. |
| Port busy / access denied | Close the TI visualizer, UniFlash or any terminal that has the ports open. On Linux, check `dialout` membership. |
| A command returns `Error` | The cfg does not match the firmware version. Use the SDK 3.x demo, or check the CLI console for the failing line. |
| Started, but no frames | The data port is wrong (it must be the "Auxiliary Data Port" at 921600 baud), or `guiMonitor` disables all outputs. |
| *Missed frames* / *parse errors* increase | The UART is overloaded: lower the frame rate or disable heatmaps. The app warns about this when you load a config. |
| Range-azimuth map looks mirrored | Tick *Mirror azimuth*. Different firmware builds order the virtual antennas differently. |

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest            # protocol, config maths, device control (via simulator), recording
```

| Module | Purpose |
|---|---|
| `awr1843/tlv.py` | binary frame protocol (parser + encoder) |
| `awr1843/config.py` | cfg parsing and radar parameter maths |
| `awr1843/device.py` | serial ports, CLI control, background reader |
| `awr1843/simulator.py` | fake board speaking the same protocol |
| `awr1843/recorder.py` | recording, export, replay |
| `awr1843/processing.py` | angle FFT and polar→cartesian heatmaps |
| `awr1843/gui/app.py` | the Qt / pyqtgraph user interface |
