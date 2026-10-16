"""A software stand-in for the AWR1843BOOST running the out-of-box demo.

It implements the same two "serial ports" as the real board: a text CLI that
understands the demo commands and a data port that emits correctly encoded TLV
frames of a synthetic scene (a person walking in a circle, an object moving
back and forth, a few static reflectors and some clutter).  Useful for trying
the app without hardware and for the automated tests.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from .config import DATA_PORT_BYTES_PER_SEC, parse_cfg
from .tlv import LOG2_Q9_TO_DB, encode_frame

KNOWN_COMMANDS = {
    "sensorStop", "sensorStart", "flushCfg", "dfeDataOutputMode", "channelCfg",
    "adcCfg", "adcbufCfg", "profileCfg", "chirpCfg", "frameCfg", "advFrameCfg",
    "subFrameCfg", "lowPower", "guiMonitor", "cfarCfg", "multiObjBeamForming",
    "clutterRemoval", "calibDcRangeSig", "extendedMaxVelocity", "bpmCfg",
    "lvdsStreamCfg", "compRangeBiasAndRxChanPhase",
    "measureRangeBiasAndRxChanPhase", "CQRxSatMonitor", "CQSigImgMonitor",
    "analogMonitor", "aoaFovCfg", "cfarFovCfg", "calibData", "version",
    "queryDemoStatus", "help", "idlePowerCycle",
}

LIVE_COMMANDS = {"clutterRemoval", "cfarCfg", "guiMonitor", "multiObjBeamForming",
                 "aoaFovCfg", "cfarFovCfg", "calibDcRangeSig", "extendedMaxVelocity"}

PROMPT = "mmwDemo:/>"


class _Pipe:
    """Thread-safe byte FIFO with a serial-port-like read API."""

    def __init__(self, timeout: float = 0.05):
        self.timeout = timeout
        self._buf = bytearray()
        self._cond = threading.Condition()
        self.closed = False

    def push(self, data: bytes) -> None:
        with self._cond:
            self._buf += data
            self._cond.notify_all()

    @property
    def in_waiting(self) -> int:
        return len(self._buf)

    def read(self, size: int = 1) -> bytes:
        with self._cond:
            if not self._buf and not self.closed:
                self._cond.wait(self.timeout)
            out = bytes(self._buf[:size])
            del self._buf[:size]
            return out

    def reset_input_buffer(self) -> None:
        with self._cond:
            self._buf.clear()

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class _CliPort(_Pipe):
    def __init__(self, radar: "SimulatedRadar"):
        super().__init__()
        self._radar = radar
        self._line = bytearray()

    def write(self, data: bytes) -> int:
        self._line += data
        while b"\n" in self._line:
            raw, _, rest = self._line.partition(b"\n")
            self._line = bytearray(rest)
            cmd = raw.decode(errors="replace").strip()
            if cmd:
                reply = self._radar.handle_command(cmd)
                self.push(f"{cmd}\r\n{reply}\r\n{PROMPT}".encode())
        return len(data)

    def close(self) -> None:
        super().close()
        self._radar.shutdown()


class _DataPort(_Pipe):
    def write(self, data: bytes) -> int:  # the data port is output only
        return len(data)


class SimulatedRadar:
    def __init__(self, seed: int | None = None, realtime: bool = True):
        self.cli_port = _CliPort(self)
        self.data_port = _DataPort()
        self.realtime = realtime
        self._rng = np.random.default_rng(seed)
        self._cfg: list[str] = []
        self._active_cfg: list[str] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_number = 0
        self._t0 = time.monotonic()
        self._sim_time = 0.0

    # ------------------------------------------------------------------ CLI
    def handle_command(self, cmd: str) -> str:
        parts = cmd.split()
        name = parts[0]
        if name not in KNOWN_COMMANDS:
            return f"'{name}' is not recognized as a CLI command\r\ntype help for more information"
        if name == "version":
            return ("Platform                : xWR18xx\r\n"
                    "mmWave SDK Version      : 03.05.00.04\r\n"
                    "Device Info             : AWR18XX (simulated)\r\n"
                    "RF F/W Version          : 02.01.00.04\r\n"
                    "Done")
        if name == "queryDemoStatus":
            running = self._thread is not None and self._thread.is_alive()
            return f"Sensor State: {2 if running else 1}\r\nData port baud rate: 921600\r\nDone"
        if name == "help":
            return "\r\n".join(sorted(KNOWN_COMMANDS)) + "\r\nDone"
        if name == "sensorStop":
            self._halt()
            return "Done"
        if name == "flushCfg":
            if self._running:
                return "Error -1"
            self._cfg = []
            return "Done"
        if name == "sensorStart":
            if self._running:
                return "Ignored: Sensor is already started\r\nDone"
            reconfig = not (len(parts) > 1 and parts[1] == "0")
            if reconfig or not self._active_cfg:
                if not any(c.startswith("profileCfg") for c in self._cfg):
                    return "Error -1\r\nmmWave Config failed"
                self._active_cfg = list(self._cfg)
            self._start()
            return "Done"
        if name == "profileCfg" and len(parts) != 15:
            return "Error: Invalid usage of the CLI command"
        if self._running:
            return "Error: Sensor is running, issue sensorStop first"
        self._cfg.append(cmd)
        if name in LIVE_COMMANDS and self._active_cfg:
            # take effect on the next "sensorStart 0" (resume without reconfig)
            key = parts[:3] if name == "cfarCfg" else parts[:1]
            self._active_cfg = [c for c in self._active_cfg
                                if c.split()[:len(key)] != key] + [cmd]
        return "Done"

    @property
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sim-radar", daemon=True)
        self._thread.start()

    def _halt(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def shutdown(self) -> None:
        self._halt()
        self.data_port.close()

    # ------------------------------------------------------------- frames
    def _run(self) -> None:
        params = parse_cfg(self._active_cfg)
        period = params.frame_period_ms / 1000.0
        next_t = time.monotonic()
        while not self._stop.is_set():
            self.data_port.push(self.make_frame(params))
            self._sim_time += period
            if self.realtime:
                next_t += period
                self._stop.wait(max(0.0, next_t - time.monotonic()))
            elif self.data_port.in_waiting > 4 << 20:
                self._stop.wait(0.01)

    def _scene(self, t: float) -> list[tuple[float, float, float, bool, int]]:
        """(x, y, z, static, num_points) of each scatterer at time t."""
        objs = []
        a = 0.45 * t
        objs.append((1.2 * math.sin(a), 3.5 + 1.2 * math.cos(a), 0.0, False, 6))  # person
        y = 5.0 + 2.5 * math.sin(0.3 * t)
        objs.append((-1.5, y, -0.3, False, 3))                                     # object
        objs.append((1.8, 5.5, 0.4, True, 2))                                      # static
        objs.append((-2.5, 7.0, 0.0, True, 2))
        objs.append((0.3, 1.5, -0.5, True, 1))
        return objs

    def make_frame(self, p) -> bytes:
        rng = self._rng
        t = self._sim_time
        dt = 0.05
        n_rb = p.num_range_bins
        n_db = p.num_doppler_bins
        r_res = p.range_resolution_m
        v_res = p.velocity_resolution_ms
        max_r = min(p.max_range_m, n_rb * r_res)
        clutter_removal = p.clutter_removal
        thr = p.cfar_range_threshold_db or 15.0

        xyzv, side = [], []
        rp_db = 52 + 4 * np.exp(-np.arange(n_rb) / 8) * 10 + rng.normal(0, 1.2, n_rb)
        rd_db = 50 + rng.normal(0, 1.5, (n_rb, n_db))
        az = (rng.normal(0, 40, (n_rb, p.num_virtual_antennas_azimuth))
              + 1j * rng.normal(0, 40, (n_rb, p.num_virtual_antennas_azimuth)))
        ant = np.arange(p.num_virtual_antennas_azimuth)

        now, later = self._scene(t), self._scene(t + dt)
        for (x, y, z, static, npts), (x2, y2, z2, _, _) in zip(now, later):
            r = math.sqrt(x * x + y * y + z * z)
            if r >= max_r:
                continue
            v = 0.0 if static else (math.sqrt(x2 * x2 + y2 * y2 + z2 * z2) - r) / dt
            v = max(-p.max_velocity_ms, min(p.max_velocity_ms - v_res, v))
            strength = 35 - 12 * math.log10(max(r, 0.5))
            rbin = min(n_rb - 1, int(round(r / r_res)))
            dbin = int(round(v / v_res)) % n_db
            rp_db[max(0, rbin - 1):rbin + 2] = np.maximum(
                rp_db[max(0, rbin - 1):rbin + 2], 52 + strength)
            rd_db[max(0, rbin - 1):rbin + 2, dbin] += strength
            if static:
                sin_t = x / math.hypot(x, y)
                steer = 2000 / max(r, 0.5) * np.exp(1j * math.pi * ant * sin_t)
                for off, gain in ((-1, 0.4), (0, 1.0), (1, 0.4)):
                    if 0 <= rbin + off < n_rb:
                        az[rbin + off] += gain * steer
                if clutter_removal:
                    continue
            if strength < thr - 5:
                continue
            for _ in range(npts):
                jitter = rng.normal(0, 0.08, 3) if npts > 1 else rng.normal(0, 0.02, 3)
                xyzv.append((x + jitter[0], y + jitter[1], z + jitter[2],
                             v + rng.normal(0, v_res / 3) if not static else 0.0))
                side.append((strength + rng.normal(0, 2), 52 + rng.normal(0, 1)))
        for _ in range(rng.poisson(1.0)):  # random clutter
            xyzv.append((rng.uniform(-3, 3), rng.uniform(0.5, max_r * 0.9),
                         rng.uniform(-1, 1), rng.normal(0, 0.5)))
            side.append((rng.uniform(8, 13), 52.0))
        if clutter_removal:
            rd_db[:, 0] = 50 + rng.normal(0, 1.5, n_rb)

        g = p.gui
        xyzv_arr = np.array(xyzv, dtype=np.float32).reshape(-1, 4) if g.detected_objects else None
        side_arr = np.array(side).reshape(-1, 2) if g.detected_objects == 1 else None
        to_raw = 1.0 / LOG2_Q9_TO_DB
        packet_args = dict(
            xyzv=xyzv_arr, snr_noise_db=side_arr,
            range_profile=np.clip(rp_db * to_raw, 0, 65535) if g.log_mag_range else None,
            noise_profile=np.clip((50 + rng.normal(0, 0.5, n_rb)) * to_raw, 0, 65535)
            if g.noise_profile else None,
            azimuth_static=az if g.range_azimuth_heatmap else None,
            range_doppler=np.clip(rd_db * to_raw, 0, 65535) if g.range_doppler_heatmap else None,
        )
        n_bytes = len(encode_frame(self._frame_number, **packet_args))
        if g.stats:
            active_us = p.active_frame_time_ms * 1000
            proc_us = 1500 + 6 * len(xyzv)
            tx_us = n_bytes / DATA_PORT_BYTES_PER_SEC * 1e6
            margin = max(0, p.frame_period_ms * 1000 - active_us - proc_us - tx_us)
            packet_args["stats"] = (int(proc_us), int(tx_us), int(margin), 12,
                                    int(30 + rng.normal(0, 2)), int(18 + rng.normal(0, 2)))
            base = 42 + 6 * (1 - math.exp(-t / 120))
            temps = [int(base + o + rng.normal(0, 0.4))
                     for o in (0, 0, 1, 1, 4, 4, 3, 1, 2, 2)]
            packet_args["temperature"] = (0, int(t * 1000), *temps)
        self._frame_number += 1
        cycles = int((t * 200e6)) & 0xFFFFFFFF
        return encode_frame(self._frame_number, cpu_cycles=cycles, **packet_args)
