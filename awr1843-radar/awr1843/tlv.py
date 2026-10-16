"""Parser (and encoder) for the UART output of the mmWave SDK 3.x demo.

Every frame the AWR1843 out-of-box demo sends on the *data* port looks like::

    +------------------- 40 byte header -------------------+
    | magic word  02 01 04 03 06 05 08 07                  |
    | version, totalPacketLen, platform, frameNumber,      |
    | timeCpuCycles, numDetectedObj, numTLVs, subFrameNum  |
    +------------------------------------------------------+
    | TLV 1: type (u32), length (u32), payload[length]      |
    | TLV 2: ...                                            |
    +------------------------------------------------------+
    | padding up to a multiple of 32 bytes                  |

All values are little-endian.  See the "mmw demo" doxygen of the SDK
(``MmwDemo_output_message_type``) for the reference definition.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_STRUCT = struct.Struct("<8sIIIIIIII")
HEADER_SIZE = HEADER_STRUCT.size  # 40
TLV_HEADER_STRUCT = struct.Struct("<II")
MAX_PACKET_SIZE = 1 << 20  # sanity limit used to re-synchronise on garbage

# The range profile / noise profile / range-Doppler heatmap are sent as
# log2(magnitude) in Q9 fixed point.  Multiply by this to get (relative) dB.
LOG2_Q9_TO_DB = 20.0 * np.log10(2.0) / 512.0


class TlvType(IntEnum):
    DETECTED_POINTS = 1
    RANGE_PROFILE = 2
    NOISE_PROFILE = 3
    AZIMUTH_STATIC_HEATMAP = 4
    RANGE_DOPPLER_HEATMAP = 5
    STATS = 6
    DETECTED_POINTS_SIDE_INFO = 7
    AZIMUTH_ELEVATION_STATIC_HEATMAP = 8
    TEMPERATURE_STATS = 9


STATS_FIELDS = (
    "inter_frame_processing_time_us",
    "transmit_output_time_us",
    "inter_frame_processing_margin_us",
    "inter_chirp_processing_margin_us",
    "active_frame_cpu_load_pct",
    "inter_frame_cpu_load_pct",
)

TEMPERATURE_SENSORS = ("rx0", "rx1", "rx2", "rx3", "tx0", "tx1", "tx2",
                       "pm", "dig0", "dig1")

POINT_DTYPE = np.dtype([
    ("x", np.float32), ("y", np.float32), ("z", np.float32),
    ("velocity", np.float32),
    ("range", np.float32), ("azimuth_deg", np.float32),
    ("elevation_deg", np.float32),
    ("snr_db", np.float32), ("noise_db", np.float32),
])


def empty_points() -> np.ndarray:
    return np.zeros(0, dtype=POINT_DTYPE)


def format_version(version: int) -> str:
    return ".".join(str((version >> s) & 0xFF) for s in (24, 16, 8, 0))


def format_platform(platform: int) -> str:
    return f"{platform:X}".replace("A", "xWR", 1) if platform else "?"


@dataclass
class Frame:
    """One decoded radar frame."""

    frame_number: int
    sub_frame: int = 0
    version: int = 0
    platform: int = 0
    cpu_cycles: int = 0
    num_detected: int = 0
    num_tlvs: int = 0
    packet_length: int = 0
    host_time: float = field(default_factory=time.time)
    points: np.ndarray = field(default_factory=empty_points)
    range_profile: np.ndarray | None = None       # uint16 log2 Q9 per range bin
    noise_profile: np.ndarray | None = None       # uint16 log2 Q9 per range bin
    azimuth_static: np.ndarray | None = None      # complex64 [range, virtual antenna]
    range_doppler: np.ndarray | None = None       # uint16 [range, doppler] (not shifted)
    stats: dict[str, int] | None = None
    temperature: dict[str, float] | None = None
    unknown_tlvs: dict[int, bytes] = field(default_factory=dict)

    @property
    def version_str(self) -> str:
        return format_version(self.version)

    @property
    def range_profile_db(self) -> np.ndarray | None:
        if self.range_profile is None:
            return None
        return self.range_profile.astype(np.float32) * LOG2_Q9_TO_DB

    @property
    def noise_profile_db(self) -> np.ndarray | None:
        if self.noise_profile is None:
            return None
        return self.noise_profile.astype(np.float32) * LOG2_Q9_TO_DB

    @property
    def range_doppler_db(self) -> np.ndarray | None:
        """Range-Doppler heatmap in dB with zero velocity in the centre."""
        if self.range_doppler is None:
            return None
        rd = self.range_doppler.astype(np.float32) * LOG2_Q9_TO_DB
        return np.fft.fftshift(rd, axes=1)

    def to_dict(self) -> dict:
        """JSON friendly per-frame metadata (no arrays)."""
        return {
            "frame_number": self.frame_number,
            "sub_frame": self.sub_frame,
            "host_time": self.host_time,
            "cpu_cycles": self.cpu_cycles,
            "num_points": int(len(self.points)),
            "packet_length": self.packet_length,
            "stats": self.stats,
            "temperature": self.temperature,
        }


class FrameParseError(ValueError):
    pass


def _points_from_xyzv(xyzv: np.ndarray) -> np.ndarray:
    pts = np.zeros(len(xyzv), dtype=POINT_DTYPE)
    if len(xyzv) == 0:
        return pts
    x, y, z, v = (xyzv[:, i] for i in range(4))
    pts["x"], pts["y"], pts["z"], pts["velocity"] = x, y, z, v
    horiz = np.hypot(x, y)
    pts["range"] = np.sqrt(horiz ** 2 + z ** 2)
    pts["azimuth_deg"] = np.degrees(np.arctan2(x, y))
    pts["elevation_deg"] = np.degrees(np.arctan2(z, horiz))
    pts["snr_db"] = np.nan
    pts["noise_db"] = np.nan
    return pts


def parse_frame(packet: bytes | bytearray | memoryview,
                host_time: float | None = None,
                num_range_bins: int | None = None) -> Frame:
    """Decode one complete packet (starting with the magic word)."""
    packet = bytes(packet)
    if len(packet) < HEADER_SIZE:
        raise FrameParseError("packet shorter than header")
    (magic, version, total_len, platform, frame_number, cpu_cycles,
     num_obj, num_tlvs, sub_frame) = HEADER_STRUCT.unpack_from(packet, 0)
    if magic != MAGIC_WORD:
        raise FrameParseError("bad magic word")
    if total_len > len(packet):
        raise FrameParseError("truncated packet")

    frame = Frame(frame_number=frame_number, sub_frame=sub_frame,
                  version=version, platform=platform, cpu_cycles=cpu_cycles,
                  num_detected=num_obj, num_tlvs=num_tlvs,
                  packet_length=total_len,
                  host_time=time.time() if host_time is None else host_time)

    side_info = None
    deferred: list[tuple[int, bytes]] = []  # heatmaps need num_range_bins
    offset = HEADER_SIZE
    for _ in range(num_tlvs):
        if offset + TLV_HEADER_STRUCT.size > total_len:
            raise FrameParseError("TLV header beyond end of packet")
        tlv_type, length = TLV_HEADER_STRUCT.unpack_from(packet, offset)
        offset += TLV_HEADER_STRUCT.size
        if offset + length > total_len:
            raise FrameParseError(f"TLV {tlv_type} length {length} beyond packet")
        payload = packet[offset:offset + length]
        offset += length

        if tlv_type == TlvType.DETECTED_POINTS:
            n = length // 16
            xyzv = np.frombuffer(payload, dtype="<f4", count=n * 4).reshape(n, 4)
            frame.points = _points_from_xyzv(xyzv)
        elif tlv_type == TlvType.DETECTED_POINTS_SIDE_INFO:
            side_info = np.frombuffer(payload, dtype="<i2",
                                      count=(length // 4) * 2).reshape(-1, 2)
        elif tlv_type == TlvType.RANGE_PROFILE:
            frame.range_profile = np.frombuffer(payload, dtype="<u2").copy()
        elif tlv_type == TlvType.NOISE_PROFILE:
            frame.noise_profile = np.frombuffer(payload, dtype="<u2").copy()
        elif tlv_type in (TlvType.AZIMUTH_STATIC_HEATMAP,
                          TlvType.RANGE_DOPPLER_HEATMAP):
            deferred.append((tlv_type, payload))
        elif tlv_type == TlvType.STATS:
            vals = struct.unpack_from(f"<{length // 4}I", payload)
            frame.stats = dict(zip(STATS_FIELDS, vals))
        elif tlv_type == TlvType.TEMPERATURE_STATS and length >= 28:
            valid, t_ms = struct.unpack_from("<iI", payload, 0)
            temps = struct.unpack_from("<10h", payload, 8)
            frame.temperature = {"valid": valid == 0, "time_ms": t_ms,
                                 **dict(zip(TEMPERATURE_SENSORS, map(float, temps)))}
        else:
            frame.unknown_tlvs[tlv_type] = payload

    if side_info is not None and len(side_info) == len(frame.points):
        frame.points["snr_db"] = side_info[:, 0] * 0.1
        frame.points["noise_db"] = side_info[:, 1] * 0.1

    if num_range_bins is None:
        for prof in (frame.range_profile, frame.noise_profile):
            if prof is not None:
                num_range_bins = len(prof)
                break
    for tlv_type, payload in deferred:
        if not num_range_bins:
            frame.unknown_tlvs[tlv_type] = payload
            continue
        if tlv_type == TlvType.AZIMUTH_STATIC_HEATMAP:
            # cmplx16ImRe_t: int16 imag, int16 real, [range bin][virtual antenna]
            raw = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
            n_ant = len(raw) // num_range_bins
            cplx = (raw[:, 1].astype(np.float32) + 1j * raw[:, 0].astype(np.float32))
            frame.azimuth_static = cplx[:n_ant * num_range_bins].reshape(
                num_range_bins, n_ant).astype(np.complex64)
        else:
            raw = np.frombuffer(payload, dtype="<u2")
            n_dop = len(raw) // num_range_bins
            frame.range_doppler = raw[:n_dop * num_range_bins].reshape(
                num_range_bins, n_dop).copy()
    return frame


class FrameStreamParser:
    """Incrementally finds and decodes frames in a byte stream.

    Feed it whatever chunks arrive from the serial port; it resynchronises on
    the magic word and silently skips garbage / truncated packets.
    """

    def __init__(self, num_range_bins: int | None = None):
        self.num_range_bins = num_range_bins
        self._buf = bytearray()
        self.frames_parsed = 0
        self.bytes_received = 0
        self.bytes_discarded = 0
        self.parse_errors = 0

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes, host_time: float | None = None) -> list[Frame]:
        self.bytes_received += len(data)
        self._buf += data
        frames: list[Frame] = []
        while True:
            start = self._buf.find(MAGIC_WORD)
            if start < 0:
                # keep a tail that might contain a partial magic word
                keep = len(MAGIC_WORD) - 1
                if len(self._buf) > keep:
                    self.bytes_discarded += len(self._buf) - keep
                    del self._buf[:-keep]
                break
            if start:
                self.bytes_discarded += start
                del self._buf[:start]
            if len(self._buf) < HEADER_SIZE:
                break
            total_len = struct.unpack_from("<I", self._buf, 12)[0]
            if total_len < HEADER_SIZE or total_len > MAX_PACKET_SIZE:
                self.parse_errors += 1
                self.bytes_discarded += 1
                del self._buf[:1]
                continue
            # A new magic word inside the packet means this one got truncated
            # (e.g. bytes lost on the UART): drop it and resync right away.
            nxt = self._buf.find(MAGIC_WORD, 1, min(total_len, len(self._buf)))
            if nxt > 0:
                self.parse_errors += 1
                self.bytes_discarded += nxt
                del self._buf[:nxt]
                continue
            if len(self._buf) < total_len:
                break
            packet = bytes(self._buf[:total_len])
            del self._buf[:total_len]
            try:
                frames.append(parse_frame(packet, host_time, self.num_range_bins))
                self.frames_parsed += 1
            except FrameParseError:
                self.parse_errors += 1
        return frames


# --------------------------------------------------------------------------
# Encoder (used by the simulator and the tests)
# --------------------------------------------------------------------------

def _tlv(tlv_type: int, payload: bytes) -> bytes:
    return TLV_HEADER_STRUCT.pack(tlv_type, len(payload)) + payload


def encode_frame(frame_number: int,
                 xyzv: np.ndarray | None = None,
                 snr_noise_db: np.ndarray | None = None,
                 range_profile: np.ndarray | None = None,
                 noise_profile: np.ndarray | None = None,
                 azimuth_static: np.ndarray | None = None,
                 range_doppler: np.ndarray | None = None,
                 stats: tuple[int, ...] | None = None,
                 temperature: tuple[int, ...] | None = None,
                 version: int = 0x03050004,
                 platform: int = 0xA1843,
                 cpu_cycles: int = 0,
                 sub_frame: int = 0) -> bytes:
    """Build a packet exactly like the firmware would."""
    tlvs = []
    n_obj = 0
    if xyzv is not None and len(xyzv):
        xyzv = np.asarray(xyzv, dtype="<f4").reshape(-1, 4)
        n_obj = len(xyzv)
        tlvs.append(_tlv(TlvType.DETECTED_POINTS, xyzv.tobytes()))
        if snr_noise_db is not None:
            side = np.round(np.asarray(snr_noise_db) * 10).astype("<i2")
            tlvs.append(_tlv(TlvType.DETECTED_POINTS_SIDE_INFO, side.tobytes()))
    if range_profile is not None:
        tlvs.append(_tlv(TlvType.RANGE_PROFILE,
                         np.asarray(range_profile).astype("<u2").tobytes()))
    if noise_profile is not None:
        tlvs.append(_tlv(TlvType.NOISE_PROFILE,
                         np.asarray(noise_profile).astype("<u2").tobytes()))
    if azimuth_static is not None:
        az = np.asarray(azimuth_static)
        imre = np.empty(az.shape + (2,), dtype="<i2")
        imre[..., 0] = np.clip(np.round(az.imag), -32768, 32767)
        imre[..., 1] = np.clip(np.round(az.real), -32768, 32767)
        tlvs.append(_tlv(TlvType.AZIMUTH_STATIC_HEATMAP, imre.tobytes()))
    if range_doppler is not None:
        tlvs.append(_tlv(TlvType.RANGE_DOPPLER_HEATMAP,
                         np.asarray(range_doppler).astype("<u2").tobytes()))
    if stats is not None:
        tlvs.append(_tlv(TlvType.STATS, struct.pack("<6I", *stats)))
    if temperature is not None:
        valid_time, temps = temperature[:2], temperature[2:]
        tlvs.append(_tlv(TlvType.TEMPERATURE_STATS,
                         struct.pack("<iI10h", *valid_time, *temps)))
    body = b"".join(tlvs)
    total = HEADER_SIZE + len(body)
    total += (-total) % 32
    header = HEADER_STRUCT.pack(MAGIC_WORD, version, total, platform,
                                frame_number, cpu_cycles, n_obj, len(tlvs),
                                sub_frame)
    packet = header + body
    return packet + b"\x0f" * (total - len(packet))
