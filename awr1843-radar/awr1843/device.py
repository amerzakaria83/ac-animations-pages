"""Talking to the AWR1843BOOST over USB.

When plugged in, the on-board XDS110 debugger exposes two virtual COM ports:

* **CLI / "Application/User UART"** (115200 baud): text commands in, text
  responses out.  This is how the radar is configured and started/stopped.
* **Data / "Auxiliary Data Port"** (921600 baud): binary TLV frames out.

The board must be running the mmWave SDK 3.x out-of-box demo
(``xwr18xx_mmw_demo.bin``) in functional mode.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

from .config import RadarParams, parse_cfg, read_cfg_lines
from .tlv import Frame, FrameStreamParser

log = logging.getLogger(__name__)

TI_VID = 0x0451
XDS110_PID = 0xBEF3
CLI_BAUD = 115_200
DATA_BAUD = 921_600

PROMPT_RE = re.compile(r"\w+:/>")
ERROR_MARKERS = ("Error", "not recognized", "Invalid usage", "Exceeded max")

FrameCallback = Callable[[Frame], None]
RawCallback = Callable[[bytes], None]


# ---------------------------------------------------------------- transport

class Transport(Protocol):
    """Minimal subset of :class:`serial.Serial` used here."""

    timeout: float | None

    def write(self, data: bytes) -> int: ...
    def read(self, size: int = 1) -> bytes: ...
    @property
    def in_waiting(self) -> int: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


def open_serial(port: str, baud: int, timeout: float = 0.05) -> Transport:
    import serial  # pyserial

    return serial.Serial(port, baud, timeout=timeout)


# ------------------------------------------------------------ port discovery

@dataclass
class PortInfo:
    device: str
    description: str
    hwid: str
    role: str = ""  # "cli", "data" or ""

    def __str__(self) -> str:
        role = f" [{self.role.upper()}]" if self.role else ""
        return f"{self.device} - {self.description}{role}"


def list_ports() -> list[PortInfo]:
    """All serial ports, with the XDS110 CLI/data ports labelled."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:  # pragma: no cover
        return []
    infos = []
    xds = []
    for p in sorted(lp.comports(), key=lambda p: p.device):
        info = PortInfo(p.device, p.description or "", p.hwid or "")
        infos.append(info)
        desc = (p.description or "").lower()
        if (p.vid == TI_VID and p.pid == XDS110_PID) or "xds110" in desc:
            xds.append((p, info))

    for p, info in xds:
        desc = (p.description or "").lower()
        if "application/user uart" in desc or "user uart" in desc:
            info.role = "cli"
        elif "auxiliary data port" in desc or "data port" in desc:
            info.role = "data"
        elif p.location and p.location.endswith(":1.0"):
            info.role = "cli"
        elif p.location and p.location.endswith(":1.3"):
            info.role = "data"
    # Fallback (e.g. macOS / Linux without descriptive names): the CLI port is
    # the first XDS110 port and the data port the second.
    if xds and not any(i.role for _, i in xds) and len(xds) >= 2:
        xds[0][1].role = "cli"
        xds[1][1].role = "data"
    return infos


def find_radar_ports() -> tuple[str | None, str | None]:
    cli = data = None
    for info in list_ports():
        if info.role == "cli" and cli is None:
            cli = info.device
        elif info.role == "data" and data is None:
            data = info.device
    return cli, data


# ------------------------------------------------------------- frame source

@dataclass
class SourceStats:
    frames: int = 0
    bytes: int = 0
    bytes_discarded: int = 0
    parse_errors: int = 0
    missed_frames: int = 0
    fps: float = 0.0
    bytes_per_sec: float = 0.0
    last_frame_number: int | None = None
    _times: deque = field(default_factory=lambda: deque(maxlen=50))
    _byte_log: deque = field(default_factory=lambda: deque(maxlen=50))

    def on_bytes(self, n: int) -> None:
        now = time.monotonic()
        self.bytes += n
        self._byte_log.append((now, self.bytes))
        if len(self._byte_log) > 1 and now - self._byte_log[0][0] > 0:
            self.bytes_per_sec = ((self.bytes - self._byte_log[0][1])
                                  / (now - self._byte_log[0][0]))

    def on_frame(self, frame: Frame) -> None:
        now = time.monotonic()
        self.frames += 1
        if (self.last_frame_number is not None
                and frame.frame_number > self.last_frame_number + 1):
            self.missed_frames += frame.frame_number - self.last_frame_number - 1
        self.last_frame_number = frame.frame_number
        self._times.append(now)
        if len(self._times) > 1:
            self.fps = (len(self._times) - 1) / max(1e-9, self._times[-1] - self._times[0])


class FrameSource:
    """Base class for anything that produces :class:`Frame` objects
    (the real radar, the simulator or a recording being replayed)."""

    def __init__(self) -> None:
        self._frame_listeners: list[FrameCallback] = []
        self._raw_listeners: list[RawCallback] = []
        self.stats = SourceStats()
        self.params: RadarParams | None = None

    def add_frame_listener(self, cb: FrameCallback) -> None:
        self._frame_listeners.append(cb)

    def remove_frame_listener(self, cb: FrameCallback) -> None:
        if cb in self._frame_listeners:
            self._frame_listeners.remove(cb)

    def add_raw_listener(self, cb: RawCallback) -> None:
        self._raw_listeners.append(cb)

    def remove_raw_listener(self, cb: RawCallback) -> None:
        if cb in self._raw_listeners:
            self._raw_listeners.remove(cb)

    def _emit_raw(self, data: bytes) -> None:
        self.stats.on_bytes(len(data))
        for cb in list(self._raw_listeners):
            try:
                cb(data)
            except Exception:  # pragma: no cover - listener bugs must not kill IO
                log.exception("raw listener failed")

    def _emit_frame(self, frame: Frame) -> None:
        self.stats.on_frame(frame)
        for cb in list(self._frame_listeners):
            try:
                cb(frame)
            except Exception:  # pragma: no cover
                log.exception("frame listener failed")

    def frames(self, timeout: float | None = None, max_queue: int = 100) -> Iterator[Frame]:
        """Blocking iterator over incoming frames (handy for scripts)."""
        q: deque[Frame] = deque(maxlen=max_queue)
        cond = threading.Condition()

        def push(frame: Frame) -> None:
            with cond:
                q.append(frame)
                cond.notify()

        self.add_frame_listener(push)
        try:
            while True:
                with cond:
                    if not q and not cond.wait_for(lambda: bool(q), timeout):
                        return
                    frame = q.popleft()
                yield frame
        finally:
            self.remove_frame_listener(push)

    def close(self) -> None:  # pragma: no cover - overridden
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------- device

@dataclass
class CommandResult:
    command: str
    ok: bool
    response: str

    def __str__(self) -> str:
        return f"{'OK ' if self.ok else 'ERR'} {self.command}: {self.response.strip()}"


class RadarError(RuntimeError):
    pass


class RadarDevice(FrameSource):
    """Controls an AWR1843 running the out-of-box demo.

    Example::

        with RadarDevice.connect() as radar:
            radar.configure(open("configs/awr1843_default.cfg").read())
            for frame in radar.frames(timeout=2):
                print(frame.frame_number, len(frame.points))
    """

    def __init__(self, cli: Transport, data: Transport,
                 cli_name: str = "", data_name: str = ""):
        super().__init__()
        self.cli = cli
        self.data = data
        self.cli_name = cli_name
        self.data_name = data_name
        self.cfg_lines: list[str] = []
        self.running = False
        self._cli_lock = threading.Lock()
        self._stop = threading.Event()
        self._parser = FrameStreamParser()
        self.cli_log: deque[str] = deque(maxlen=2000)
        self.cli_listeners: list[Callable[[str], None]] = []
        self._reader = threading.Thread(target=self._read_loop,
                                        name="awr1843-data", daemon=True)
        self._reader.start()

    # ---------------------------------------------------------- construction
    @classmethod
    def connect(cls, cli_port: str | None = None, data_port: str | None = None,
                cli_baud: int = CLI_BAUD, data_baud: int = DATA_BAUD) -> "RadarDevice":
        if cli_port is None or data_port is None:
            auto_cli, auto_data = find_radar_ports()
            cli_port = cli_port or auto_cli
            data_port = data_port or auto_data
        if not cli_port or not data_port:
            raise RadarError(
                "Could not find the AWR1843 serial ports automatically. "
                "Pass them explicitly (run `python -m awr1843 ports` to list them).")
        cli = open_serial(cli_port, cli_baud, timeout=0.05)
        try:
            data = open_serial(data_port, data_baud, timeout=0.05)
        except Exception:
            cli.close()
            raise
        return cls(cli, data, cli_port, data_port)

    @classmethod
    def simulated(cls, **kwargs) -> "RadarDevice":
        from .simulator import SimulatedRadar

        sim = SimulatedRadar(**kwargs)
        return cls(sim.cli_port, sim.data_port, "SIM-CLI", "SIM-DATA")

    def close(self) -> None:
        if self.running:
            try:
                self.stop()
            except Exception:
                pass
        self._stop.set()
        self._reader.join(timeout=1.0)
        for port in (self.cli, self.data):
            try:
                port.close()
            except Exception:
                pass

    # ------------------------------------------------------------------- CLI
    def _log_cli(self, text: str) -> None:
        for line in text.splitlines():
            self.cli_log.append(line)
        for cb in list(self.cli_listeners):
            try:
                cb(text)
            except Exception:  # pragma: no cover
                log.exception("cli listener failed")

    def send_command(self, command: str, timeout: float = 2.0) -> CommandResult:
        """Send one CLI command and wait for the ``mmwDemo:/>`` prompt."""
        command = command.strip()
        with self._cli_lock:
            # discard anything unsolicited
            try:
                stale = self.cli.read(self.cli.in_waiting or 0)
                if stale:
                    self._log_cli(stale.decode(errors="replace"))
            except Exception:
                pass
            self.cli.write((command + "\n").encode())
            self._log_cli(f"> {command}")
            buf = ""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                chunk = self.cli.read(self.cli.in_waiting or 1)
                if chunk:
                    buf += chunk.decode(errors="replace")
                    body = buf.split("\n", 1)[1] if "\n" in buf else ""
                    if PROMPT_RE.search(body):
                        break
            # strip the echoed command and the prompt
            lines = [ln.strip() for ln in buf.replace("\r", "").split("\n")]
            lines = [ln for ln in lines if ln and ln != command
                     and not PROMPT_RE.fullmatch(ln)]
            lines = [PROMPT_RE.sub("", ln).strip() for ln in lines]
            lines = [ln for ln in lines if ln and ln != command]
            response = "\n".join(lines)
            if not buf:
                ok = False
                response = "(no response - is the board powered and running the demo firmware?)"
            else:
                ok = not any(m in response for m in ERROR_MARKERS)
            self._log_cli(response)
            return CommandResult(command, ok, response)

    def version(self) -> str:
        return self.send_command("version").response

    def configure(self, cfg: str | list[str], start: bool = True,
                  progress: Callable[[int, int, CommandResult], None] | None = None,
                  stop_on_error: bool = True) -> list[CommandResult]:
        """Send a full configuration.  ``sensorStart`` is sent last if
        ``start`` is true (any ``sensorStart`` line in the cfg is handled too)."""
        lines = read_cfg_lines(cfg) if isinstance(cfg, str) else list(cfg)
        body = [ln for ln in lines if not ln.startswith("sensorStart")]
        if not body or body[0] != "sensorStop":
            body.insert(0, "sensorStop")
        if "flushCfg" not in body:
            body.insert(1, "flushCfg")
        if start:
            body.append("sensorStart")
        self.cfg_lines = lines
        self.params = parse_cfg(lines)
        self._parser = FrameStreamParser(self.params.num_range_bins)
        self.running = False
        results = []
        for i, line in enumerate(body):
            slow = line.startswith(("sensorStart", "sensorStop", "calib"))
            res = self.send_command(line, timeout=5.0 if slow else 2.0)
            # sensorStop on an idle sensor reports an error on some builds; harmless
            if line == "sensorStop":
                res.ok = True
            results.append(res)
            if progress:
                progress(i + 1, len(body), res)
            if not res.ok and stop_on_error:
                raise RadarError(f"Command failed: {res}")
        if start:
            self.running = True
            self.stats.last_frame_number = None
        return results

    def start(self, reconfigure: bool = False) -> CommandResult:
        """Start (``sensorStart 0`` resumes without re-applying the config)."""
        res = self.send_command("sensorStart" if reconfigure else "sensorStart 0",
                                timeout=5.0)
        if res.ok:
            self.running = True
            self.stats.last_frame_number = None
        return res

    def stop(self) -> CommandResult:
        res = self.send_command("sensorStop", timeout=5.0)
        self.running = False
        return res

    def apply_live(self, commands: list[str]) -> list[CommandResult]:
        """Stop, send some commands (e.g. ``clutterRemoval``, ``cfarCfg``) and
        resume without a full reconfiguration."""
        was_running = self.running
        results = [self.stop()] if was_running else []
        for cmd in commands:
            results.append(self.send_command(cmd))
            self._remember(cmd)
        if was_running:
            results.append(self.start())
        return results

    def _remember(self, cmd: str) -> None:
        """Keep ``cfg_lines`` in sync with commands changed at runtime."""
        parts = cmd.split()
        if not parts:
            return
        key = parts[:3] if parts[0] == "cfarCfg" else parts[:1]
        for i, line in enumerate(self.cfg_lines):
            lp = line.split()
            if lp[:len(key)] == key:
                self.cfg_lines[i] = cmd
                break
        else:
            self.cfg_lines.append(cmd)
        self.params = parse_cfg(self.cfg_lines)

    # ------------------------------------------------------------ data port
    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self.data.read(max(1, min(self.data.in_waiting, 65536)))
            except Exception as exc:  # port unplugged etc.
                if not self._stop.is_set():
                    log.error("data port read failed: %s", exc)
                    self._log_cli(f"[data port error: {exc}]")
                break
            if not chunk:
                continue
            now = time.time()
            self._emit_raw(chunk)
            for frame in self._parser.feed(chunk, now):
                self._emit_frame(frame)
            self.stats.bytes_discarded = self._parser.bytes_discarded
            self.stats.parse_errors = self._parser.parse_errors
