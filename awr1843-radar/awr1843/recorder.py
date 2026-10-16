"""Recording live data to disk, exporting raw captures, and replaying them.

A recording folder contains:

* ``raw_data.bin``     - every byte received on the data port (lossless, replayable)
* ``config.cfg``       - the configuration that was used
* ``points.csv``       - one row per detected point
* ``telemetry.csv``    - one row per frame: point count, CPU stats, temperatures
* ``frames.jsonl``     - the same per-frame metadata as JSON lines
* ``range_profile.csv`` / ``noise_profile.csv`` - one row per frame, dB per range bin
* ``heatmaps.npz``     - range-Doppler (dB) and azimuth-static (complex) heatmaps
"""

from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import RadarParams, load_cfg_file, parse_cfg
from .device import FrameSource
from .tlv import STATS_FIELDS, TEMPERATURE_SENSORS, Frame, FrameStreamParser

POINT_COLUMNS = ["frame", "host_time", "sub_frame", "x", "y", "z", "velocity",
                 "range", "azimuth_deg", "elevation_deg", "snr_db", "noise_db"]
TELEMETRY_COLUMNS = (["frame", "host_time", "num_points", "cpu_cycles"]
                     + list(STATS_FIELDS) + [f"temp_{s}" for s in TEMPERATURE_SENSORS])


def new_recording_dir(base: str | Path = "recordings") -> Path:
    path = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


class Recorder:
    def __init__(self, out_dir: str | Path, cfg_lines: list[str] | None = None,
                 write_raw: bool = True, save_heatmaps: bool = True,
                 max_heatmap_frames: int = 20_000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.closed = False
        self.frames = 0
        self.points = 0
        self.bytes = 0
        self.started = time.time()
        self.save_heatmaps = save_heatmaps
        self.max_heatmap_frames = max_heatmap_frames
        self._heatmap_frames: list[int] = []
        self._rd: list[np.ndarray] = []
        self._az: list[np.ndarray] = []

        if cfg_lines:
            (self.out_dir / "config.cfg").write_text("\n".join(cfg_lines) + "\n")
        self._raw = open(self.out_dir / "raw_data.bin", "wb") if write_raw else None
        self._points_f = open(self.out_dir / "points.csv", "w", newline="")
        self._points = csv.writer(self._points_f)
        self._points.writerow(POINT_COLUMNS)
        self._tele_f = open(self.out_dir / "telemetry.csv", "w", newline="")
        self._tele = csv.writer(self._tele_f)
        self._tele.writerow(TELEMETRY_COLUMNS)
        self._jsonl = open(self.out_dir / "frames.jsonl", "w")
        self._rp_f = self._np_f = None
        self._rp = self._np = None

    # Callbacks matching FrameSource listeners --------------------------------
    def on_raw(self, data: bytes) -> None:
        with self._lock:
            if self._raw and not self.closed:
                self._raw.write(data)
                self.bytes += len(data)

    def on_frame(self, f: Frame) -> None:
        with self._lock:
            if self.closed:
                return
            self.frames += 1
            for p in f.points:
                self._points.writerow([f.frame_number, f"{f.host_time:.4f}", f.sub_frame]
                                      + [f"{float(p[c]):.4f}" for c in POINT_COLUMNS[3:]])
            self.points += len(f.points)
            stats = f.stats or {}
            temp = f.temperature or {}
            self._tele.writerow([f.frame_number, f"{f.host_time:.4f}", len(f.points),
                                 f.cpu_cycles]
                                + [stats.get(k, "") for k in STATS_FIELDS]
                                + [temp.get(s, "") for s in TEMPERATURE_SENSORS])
            self._jsonl.write(json.dumps(f.to_dict()) + "\n")
            if f.range_profile is not None:
                if self._rp is None:
                    self._rp_f = open(self.out_dir / "range_profile.csv", "w", newline="")
                    self._rp = csv.writer(self._rp_f)
                    self._rp.writerow(["frame", "host_time"]
                                      + [f"bin{i}" for i in range(len(f.range_profile))])
                self._rp.writerow([f.frame_number, f"{f.host_time:.4f}"]
                                  + [f"{v:.2f}" for v in f.range_profile_db])
            if f.noise_profile is not None:
                if self._np is None:
                    self._np_f = open(self.out_dir / "noise_profile.csv", "w", newline="")
                    self._np = csv.writer(self._np_f)
                    self._np.writerow(["frame", "host_time"]
                                      + [f"bin{i}" for i in range(len(f.noise_profile))])
                self._np.writerow([f.frame_number, f"{f.host_time:.4f}"]
                                  + [f"{v:.2f}" for v in f.noise_profile_db])
            if (self.save_heatmaps and len(self._heatmap_frames) < self.max_heatmap_frames
                    and (f.range_doppler is not None or f.azimuth_static is not None)):
                self._heatmap_frames.append(f.frame_number)
                if f.range_doppler is not None:
                    self._rd.append(f.range_doppler_db.astype(np.float16))
                if f.azimuth_static is not None:
                    self._az.append(f.azimuth_static)

    def attach(self, source: FrameSource) -> None:
        source.add_raw_listener(self.on_raw)
        source.add_frame_listener(self.on_frame)

    def detach(self, source: FrameSource) -> None:
        source.remove_raw_listener(self.on_raw)
        source.remove_frame_listener(self.on_frame)

    def close(self) -> dict:
        with self._lock:
            if self.closed:
                return self.summary()
            self.closed = True
            for fh in (self._raw, self._points_f, self._tele_f, self._jsonl,
                       self._rp_f, self._np_f):
                if fh:
                    fh.close()
            if self._heatmap_frames:
                arrays = {"frame_numbers": np.array(self._heatmap_frames)}
                if self._rd and len(self._rd) == len(self._heatmap_frames):
                    arrays["range_doppler_db"] = np.stack(self._rd)
                if self._az and len(self._az) == len(self._heatmap_frames):
                    arrays["azimuth_static"] = np.stack(self._az)
                np.savez_compressed(self.out_dir / "heatmaps.npz", **arrays)
            self._rd, self._az = [], []
        return self.summary()

    def summary(self) -> dict:
        return {"dir": str(self.out_dir), "frames": self.frames, "points": self.points,
                "bytes": self.bytes, "seconds": time.time() - self.started}


def read_raw_frames(raw_path: str | Path, num_range_bins: int | None = None,
                    chunk: int = 1 << 16):
    """Yield all frames contained in a ``raw_data.bin`` capture."""
    parser = FrameStreamParser(num_range_bins)
    with open(raw_path, "rb") as fh:
        while data := fh.read(chunk):
            yield from parser.feed(data, host_time=0.0)


def _params_for(raw_path: Path, cfg_path: str | Path | None) -> tuple[RadarParams | None, list[str]]:
    cfg = Path(cfg_path) if cfg_path else raw_path.with_name("config.cfg")
    if cfg.exists():
        lines = load_cfg_file(cfg)
        return parse_cfg(lines), lines
    return None, []


def export_raw(raw_path: str | Path, out_dir: str | Path,
               cfg_path: str | Path | None = None) -> dict:
    """Convert a raw capture into CSV / JSON / NPZ files."""
    raw_path = Path(raw_path)
    params, lines = _params_for(raw_path, cfg_path)
    rec = Recorder(out_dir, lines, write_raw=False)
    period = params.frame_period_ms / 1000 if params else 0.1
    for i, frame in enumerate(read_raw_frames(
            raw_path, params.num_range_bins if params else None)):
        frame.host_time = i * period
        rec.on_frame(frame)
    return rec.close()


class ReplaySource(FrameSource):
    """Plays a ``raw_data.bin`` capture back as if it came from the radar."""

    def __init__(self, raw_path: str | Path, cfg_path: str | Path | None = None,
                 speed: float = 1.0, loop: bool = True):
        super().__init__()
        self.raw_path = Path(raw_path)
        self.params, self.cfg_lines = _params_for(self.raw_path, cfg_path)
        self.speed = speed
        self.loop = loop
        self.paused = False
        self.running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        times = []
        jsonl = self.raw_path.with_name("frames.jsonl")
        if jsonl.exists():
            with open(jsonl) as fh:
                times = [json.loads(line).get("host_time", 0.0) for line in fh if line.strip()]
        self._frame_times = times

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, name="replay", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.running = False

    def _run(self) -> None:
        default_period = self.params.frame_period_ms / 1000 if self.params else 0.1
        nrb = self.params.num_range_bins if self.params else None
        while not self._stop.is_set():
            for i, frame in enumerate(read_raw_frames(self.raw_path, nrb)):
                if self._stop.is_set():
                    break
                while self.paused and not self._stop.is_set():
                    time.sleep(0.05)
                if i + 1 < len(self._frame_times) and self._frame_times[i] > 0:
                    period = self._frame_times[i + 1] - self._frame_times[i]
                    period = min(max(period, 0.0), 2.0)
                else:
                    period = default_period
                frame.host_time = time.time()
                self.stats.on_bytes(frame.packet_length)
                self._emit_frame(frame)
                self._stop.wait(period / max(self.speed, 1e-3))
            if not self.loop:
                break
            self.stats.last_frame_number = None
        self.running = False
