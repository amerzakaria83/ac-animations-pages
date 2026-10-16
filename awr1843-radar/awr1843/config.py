"""Parsing of mmWave SDK ``.cfg`` files and derivation of radar parameters.

A ``.cfg`` file is simply the list of CLI commands that is sent, line by line,
to the out-of-box demo running on the AWR1843.  From the ``channelCfg``,
``profileCfg``, ``chirpCfg`` and ``frameCfg`` commands we can derive everything
needed to interpret the data the radar sends back (range/velocity per bin,
number of bins, frame rate, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

SPEED_OF_LIGHT = 299_792_458.0  # m/s

# UART "Auxiliary Data Port" of the XDS110: 921600 baud, 8N1 -> 10 bits/byte.
DATA_PORT_BAUD = 921_600
DATA_PORT_BYTES_PER_SEC = DATA_PORT_BAUD / 10

# Sizes of the TLV payloads the SDK 3.x demo sends (bytes).
FRAME_HEADER_BYTES = 40
TLV_HEADER_BYTES = 8
POINT_BYTES = 16          # x, y, z, velocity as float32
POINT_SIDE_INFO_BYTES = 4  # snr, noise as int16
STATS_BYTES = 24
TEMPERATURE_BYTES = 28


def _next_pow2(n: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1, n))))


def _popcount(n: int) -> int:
    return bin(n).count("1")


def read_cfg_lines(text: str) -> list[str]:
    """Return the non-empty, non-comment command lines of a cfg file."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def load_cfg_file(path: str | Path) -> list[str]:
    return read_cfg_lines(Path(path).read_text(encoding="utf-8", errors="replace"))


@dataclass
class GuiMonitor:
    """Which TLVs the demo is asked to send (``guiMonitor`` command, SDK 3.x)."""

    detected_objects: int = 1  # 0 = off, 1 = points + side info, 2 = points only
    log_mag_range: bool = True
    noise_profile: bool = False
    range_azimuth_heatmap: bool = False
    range_doppler_heatmap: bool = False
    stats: bool = True


@dataclass
class RadarParams:
    """Radar parameters derived from a configuration."""

    rx_mask: int = 0b1111
    tx_mask: int = 0b111
    start_freq_ghz: float = 77.0
    idle_time_us: float = 7.0
    adc_start_time_us: float = 7.0
    ramp_end_time_us: float = 57.14
    freq_slope_mhz_us: float = 70.0
    num_adc_samples: int = 256
    sample_rate_ksps: float = 5209.0
    rx_gain_db: int = 30
    chirp_start: int = 0
    chirp_end: int = 2
    num_loops: int = 16
    num_frames: int = 0
    frame_period_ms: float = 100.0
    chirp_tx_masks: dict[int, int] = field(default_factory=dict)
    gui: GuiMonitor = field(default_factory=GuiMonitor)
    clutter_removal: bool = False
    cfar_range_threshold_db: float | None = None
    cfar_doppler_threshold_db: float | None = None
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ counts
    @property
    def num_rx(self) -> int:
        return _popcount(self.rx_mask & 0xF)

    @property
    def chirps_per_loop(self) -> int:
        return self.chirp_end - self.chirp_start + 1

    @property
    def num_tx(self) -> int:
        """Number of TDM transmit slots (one chirp per TX in the loop)."""
        return max(1, self.chirps_per_loop)

    @property
    def tx_used_mask(self) -> int:
        mask = 0
        for idx in range(self.chirp_start, self.chirp_end + 1):
            mask |= self.chirp_tx_masks.get(idx, 0)
        return mask or self.tx_mask

    @property
    def num_tx_azimuth(self) -> int:
        # On the AWR1843BOOST, TX1 (bit 0) and TX3 (bit 2) are in the azimuth
        # plane, TX2 (bit 1) is offset in elevation.
        return _popcount(self.tx_used_mask & 0b101)

    @property
    def num_tx_elevation(self) -> int:
        return _popcount(self.tx_used_mask & 0b010)

    @property
    def num_virtual_antennas(self) -> int:
        return self.num_tx * self.num_rx

    @property
    def num_virtual_antennas_azimuth(self) -> int:
        return max(1, self.num_tx_azimuth) * self.num_rx

    @property
    def num_range_bins(self) -> int:
        return _next_pow2(self.num_adc_samples)

    @property
    def num_doppler_bins(self) -> int:
        return _next_pow2(self.num_loops)

    # ---------------------------------------------------------------- physics
    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / (self.start_freq_ghz * 1e9)

    @property
    def chirp_time_us(self) -> float:
        return self.idle_time_us + self.ramp_end_time_us

    @property
    def sampled_bandwidth_mhz(self) -> float:
        sampling_time_us = self.num_adc_samples / (self.sample_rate_ksps / 1e3)
        return self.freq_slope_mhz_us * sampling_time_us

    @property
    def range_resolution_m(self) -> float:
        """Distance covered by one range bin."""
        fs = self.sample_rate_ksps * 1e3
        slope = self.freq_slope_mhz_us * 1e12
        return SPEED_OF_LIGHT * fs / (2 * slope * self.num_range_bins)

    @property
    def max_range_m(self) -> float:
        """IF-bandwidth limited maximum range (90 % of the sampling rate)."""
        fs = self.sample_rate_ksps * 1e3
        slope = self.freq_slope_mhz_us * 1e12
        return 0.9 * SPEED_OF_LIGHT * fs / (2 * slope)

    @property
    def velocity_resolution_ms(self) -> float:
        tc = self.chirp_time_us * 1e-6
        return self.wavelength_m / (2 * self.num_doppler_bins * self.num_tx * tc)

    @property
    def max_velocity_ms(self) -> float:
        tc = self.chirp_time_us * 1e-6
        return self.wavelength_m / (4 * self.num_tx * tc)

    @property
    def frame_rate_hz(self) -> float:
        return 1000.0 / self.frame_period_ms if self.frame_period_ms > 0 else 0.0

    @property
    def active_frame_time_ms(self) -> float:
        return self.num_loops * self.chirps_per_loop * self.chirp_time_us / 1000.0

    # ---------------------------------------------------------------- axes
    def range_axis(self, num_bins: int | None = None):
        import numpy as np

        n = num_bins or self.num_range_bins
        return np.arange(n) * self.range_resolution_m

    def doppler_axis(self, num_bins: int | None = None):
        """Velocity (m/s) for each Doppler bin *after* fftshift."""
        import numpy as np

        n = num_bins or self.num_doppler_bins
        return (np.arange(n) - n // 2) * self.velocity_resolution_ms

    # ---------------------------------------------------------- bandwidth
    def estimated_frame_bytes(self, num_points: int = 30) -> int:
        g = self.gui
        n = FRAME_HEADER_BYTES
        if g.detected_objects:
            n += TLV_HEADER_BYTES + POINT_BYTES * num_points
            if g.detected_objects == 1:
                n += TLV_HEADER_BYTES + POINT_SIDE_INFO_BYTES * num_points
        if g.log_mag_range:
            n += TLV_HEADER_BYTES + 2 * self.num_range_bins
        if g.noise_profile:
            n += TLV_HEADER_BYTES + 2 * self.num_range_bins
        if g.range_azimuth_heatmap:
            n += TLV_HEADER_BYTES + 4 * self.num_range_bins * self.num_virtual_antennas_azimuth
        if g.range_doppler_heatmap:
            n += TLV_HEADER_BYTES + 2 * self.num_range_bins * self.num_doppler_bins
        if g.stats:
            n += TLV_HEADER_BYTES + STATS_BYTES + TLV_HEADER_BYTES + TEMPERATURE_BYTES
        return int(math.ceil(n / 32) * 32)  # packets are padded to 32 bytes

    def estimated_data_rate(self, num_points: int = 30) -> float:
        """Bytes per second on the data port."""
        return self.estimated_frame_bytes(num_points) * self.frame_rate_hz

    def data_port_load(self, num_points: int = 30) -> float:
        """Fraction (0..1+) of the data UART bandwidth that will be used."""
        return self.estimated_data_rate(num_points) / DATA_PORT_BYTES_PER_SEC

    def summary(self) -> dict[str, str]:
        return {
            "Antennas": f"{self.num_tx} TX x {self.num_rx} RX "
                        f"({self.num_virtual_antennas} virtual)",
            "Start frequency": f"{self.start_freq_ghz:.2f} GHz",
            "Sampled bandwidth": f"{self.sampled_bandwidth_mhz:.0f} MHz",
            "Range resolution": f"{self.range_resolution_m * 100:.1f} cm",
            "Max range": f"{self.max_range_m:.2f} m",
            "Velocity resolution": f"{self.velocity_resolution_ms:.3f} m/s",
            "Max velocity": f"±{self.max_velocity_ms:.2f} m/s",
            "Range bins": str(self.num_range_bins),
            "Doppler bins": str(self.num_doppler_bins),
            "Frame rate": f"{self.frame_rate_hz:.1f} Hz",
            "Active chirp time": f"{self.active_frame_time_ms:.2f} ms / "
                                 f"{self.frame_period_ms:.0f} ms",
            "Data port load": f"{self.data_port_load() * 100:.0f} % "
                              f"(~{self.estimated_frame_bytes()} B/frame)",
        }


def parse_cfg(lines: list[str] | str) -> RadarParams:
    """Derive :class:`RadarParams` from cfg lines (or the cfg file text)."""
    if isinstance(lines, str):
        lines = read_cfg_lines(lines)
    p = RadarParams()
    profile_seen = False
    for line in lines:
        parts = line.split()
        cmd, args = parts[0], parts[1:]
        try:
            if cmd == "dfeDataOutputMode" and args and args[0] != "1":
                p.warnings.append(
                    "Only dfeDataOutputMode 1 (frame based chirps) is fully "
                    "supported; derived values may be wrong for advanced frames.")
            elif cmd == "channelCfg":
                p.rx_mask = int(args[0])
                p.tx_mask = int(args[1])
            elif cmd == "profileCfg" and not profile_seen:
                profile_seen = True
                p.start_freq_ghz = float(args[1])
                p.idle_time_us = float(args[2])
                p.adc_start_time_us = float(args[3])
                p.ramp_end_time_us = float(args[4])
                p.freq_slope_mhz_us = float(args[7])
                p.num_adc_samples = int(args[9])
                p.sample_rate_ksps = float(args[10])
                if len(args) > 13:
                    p.rx_gain_db = int(args[13])
            elif cmd == "chirpCfg":
                start, end = int(args[0]), int(args[1])
                for idx in range(start, end + 1):
                    p.chirp_tx_masks[idx] = int(args[7])
            elif cmd == "frameCfg":
                p.chirp_start = int(args[0])
                p.chirp_end = int(args[1])
                p.num_loops = int(args[2])
                p.num_frames = int(args[3])
                p.frame_period_ms = float(args[4])
            elif cmd == "guiMonitor":
                vals = [int(float(a)) for a in args[1:7]]
                vals += [0] * (6 - len(vals))
                p.gui = GuiMonitor(vals[0], bool(vals[1]), bool(vals[2]),
                                   bool(vals[3]), bool(vals[4]), bool(vals[5]))
            elif cmd == "clutterRemoval":
                p.clutter_removal = bool(int(args[1]))
            elif cmd == "cfarCfg":
                # cfarCfg <subFrame> <procDirection> <mode> <noiseWin> <guardLen>
                #         <divShift> <cyclicMode> <thresholdScale dB> <peakGrouping>
                thr = float(args[7])
                if args[1] == "0":
                    p.cfar_range_threshold_db = thr
                else:
                    p.cfar_doppler_threshold_db = thr
        except (IndexError, ValueError):
            p.warnings.append(f"Could not parse line: {line!r}")
    if not profile_seen:
        p.warnings.append("No profileCfg found; using default chirp parameters.")
    if p.active_frame_time_ms > p.frame_period_ms:
        p.warnings.append(
            f"Active chirp time ({p.active_frame_time_ms:.1f} ms) exceeds the "
            f"frame period ({p.frame_period_ms:.0f} ms).")
    load = p.data_port_load()
    if load > 0.85:
        p.warnings.append(
            f"Estimated data port load is {load * 100:.0f} %: the UART cannot keep "
            "up. Increase the frame period or disable heatmaps in guiMonitor.")
    return p
