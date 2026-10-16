import struct
import time

import numpy as np
import pytest

from awr1843.config import load_cfg_file, parse_cfg
from awr1843.device import RadarDevice
from awr1843.processing import CartesianMapper, range_azimuth_map
from awr1843.recorder import Recorder, ReplaySource, export_raw, read_raw_frames
from awr1843.tlv import (LOG2_Q9_TO_DB, MAGIC_WORD, FrameStreamParser, TlvType,
                         encode_frame, parse_frame)
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parents[1] / "awr1843" / "configs"


# ------------------------------------------------------------------ config
def test_sdk_profile_matches_ti_visualizer_numbers():
    p = parse_cfg(load_cfg_file(CFG_DIR / "awr1843_sdk_profile_3d.cfg"))
    # TI's visualizer reports 0.044 m range resolution, 1 m/s max velocity,
    # 0.13 m/s velocity resolution for this profile.
    assert p.range_resolution_m == pytest.approx(0.0436, abs=5e-4)
    assert p.max_velocity_ms == pytest.approx(1.0, abs=0.02)
    assert p.velocity_resolution_ms == pytest.approx(0.125, abs=0.005)
    assert (p.num_tx, p.num_rx, p.num_range_bins, p.num_doppler_bins) == (3, 4, 256, 16)
    assert p.num_tx_azimuth == 2 and p.num_tx_elevation == 1
    assert p.num_virtual_antennas_azimuth == 8
    assert p.frame_rate_hz == 10
    assert not p.warnings


@pytest.mark.parametrize("name", ["awr1843_default.cfg", "awr1843_heatmaps.cfg"])
def test_bundled_configs_are_consistent(name):
    p = parse_cfg(load_cfg_file(CFG_DIR / name))
    assert not p.warnings, p.warnings
    assert p.data_port_load() < 0.85


def test_overloaded_uart_warns():
    lines = load_cfg_file(CFG_DIR / "awr1843_heatmaps.cfg")
    lines = [l.replace("frameCfg 0 2 16 0 300", "frameCfg 0 2 16 0 100") for l in lines]
    assert any("UART" in w for w in parse_cfg(lines).warnings)


# --------------------------------------------------------------------- tlv
def _sample_packet(n_rb=64, n_db=16):
    xyzv = np.array([[1.0, 2.0, 0.5, -0.25], [-0.5, 4.0, 0.0, 0.0]], np.float32)
    rp = np.arange(n_rb) * 10
    rd = np.arange(n_rb * n_db).reshape(n_rb, n_db)
    az = (np.arange(n_rb * 8).reshape(n_rb, 8) + 1j * 3).astype(np.complex64)
    return encode_frame(42, xyzv=xyzv, snr_noise_db=np.array([[20.5, 50.0], [12.0, 49.9]]),
                        range_profile=rp, noise_profile=rp // 2, azimuth_static=az,
                        range_doppler=rd, stats=(1, 2, 3, 4, 5, 6),
                        temperature=(0, 1234, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)), \
        xyzv, rp, rd, az


def test_encode_parse_roundtrip():
    packet, xyzv, rp, rd, az = _sample_packet()
    assert packet.startswith(MAGIC_WORD) and len(packet) % 32 == 0
    f = parse_frame(packet)
    assert f.frame_number == 42 and f.platform == 0xA1843 and f.version_str == "3.5.0.4"
    assert f.num_detected == 2 and len(f.points) == 2
    np.testing.assert_allclose(f.points["x"], xyzv[:, 0])
    np.testing.assert_allclose(f.points["velocity"], xyzv[:, 3])
    assert f.points["range"][0] == pytest.approx(np.sqrt(1 + 4 + 0.25))
    assert f.points["azimuth_deg"][0] == pytest.approx(np.degrees(np.arctan2(1, 2)))
    np.testing.assert_allclose(f.points["snr_db"], [20.5, 12.0], atol=1e-4)
    np.testing.assert_array_equal(f.range_profile, rp)
    assert f.range_profile_db[1] == pytest.approx(10 * LOG2_Q9_TO_DB)
    np.testing.assert_array_equal(f.range_doppler, rd)
    assert f.range_doppler_db.shape == rd.shape
    np.testing.assert_allclose(f.azimuth_static, az)
    assert f.stats["active_frame_cpu_load_pct"] == 5
    assert f.temperature["rx0"] == 40 and f.temperature["dig1"] == 49
    assert f.temperature["valid"]


def test_stream_parser_handles_chunking_garbage_and_truncation():
    p1, *_ = _sample_packet()
    p2 = encode_frame(43)
    truncated = p1[:100]
    stream = b"junk\x02\x01" + p1 + truncated + p2 + b"\x02\x01\x04"
    parser = FrameStreamParser()
    frames = []
    for i in range(0, len(stream), 7):  # feed in awkward 7 byte chunks
        frames += parser.feed(stream[i:i + 7])
    assert [f.frame_number for f in frames] == [42, 43]
    assert parser.parse_errors == 1


def test_stream_parser_rejects_bogus_length():
    bad = bytearray(encode_frame(1))
    struct.pack_into("<I", bad, 12, 10)  # length smaller than the header
    parser = FrameStreamParser()
    assert parser.feed(bytes(bad) + encode_frame(2))[0].frame_number == 2


# -------------------------------------------------------------- processing
def test_range_azimuth_peak_lands_at_target_angle():
    n_rb, sin_t = 32, 0.5
    az = np.zeros((n_rb, 8), np.complex64)
    az[10] = np.exp(1j * np.pi * np.arange(8) * sin_t)
    ra = range_azimuth_map(az, 64)
    assert np.unravel_index(ra.argmax(), ra.shape) == (10, 32 + 16)
    mapper = CartesianMapper(n_rb, 0.1, 64, grid=128)
    img = mapper(ra)
    x, y = np.unravel_index(img.argmax(), img.shape)
    xs = np.linspace(-mapper.max_range, mapper.max_range, 128)
    assert xs[x] > 0  # target on the right-hand side


# ------------------------------------------------- device + simulator e2e
@pytest.fixture
def radar():
    dev = RadarDevice.simulated(seed=1)
    yield dev
    dev.close()


def test_cli_commands(radar):
    assert "03.05" in radar.version()
    res = radar.send_command("notACommand")
    assert not res.ok and "not recognized" in res.response


def test_configure_and_stream(radar):
    cfg = load_cfg_file(CFG_DIR / "awr1843_heatmaps.cfg")
    results = radar.configure(cfg)
    assert all(r.ok for r in results)
    assert radar.running and radar.params.frame_period_ms == 300
    frames = []
    for f in radar.frames(timeout=3):
        frames.append(f)
        if len(frames) == 3:
            break
    assert len(frames) == 3
    f = frames[-1]
    assert len(f.points) > 3
    assert f.range_profile.shape == (256,)
    assert f.range_doppler.shape == (256, 16)
    assert f.azimuth_static.shape == (256, 8)
    assert f.stats and f.temperature
    radar.stop()
    assert not radar.running


def test_bad_config_raises(radar):
    from awr1843.device import RadarError

    with pytest.raises(RadarError):
        radar.configure(["profileCfg 0 77"])


def test_apply_live_clutter_removal_removes_static_points(radar):
    radar.configure(load_cfg_file(CFG_DIR / "awr1843_default.cfg"))
    next(radar.frames(timeout=2))
    results = radar.apply_live(["clutterRemoval -1 1"])
    assert all(r.ok for r in results) and radar.running
    assert radar.params.clutter_removal
    it = radar.frames(timeout=2)
    next(it)
    f = next(it)
    assert f.points is not None
    # the simulator's static reflectors have exactly zero velocity
    assert not np.any(f.points["velocity"] == 0.0)


# ------------------------------------------------- recording and replay
def test_record_export_replay(tmp_path, radar):
    cfg = load_cfg_file(CFG_DIR / "awr1843_default.cfg")
    rec = Recorder(tmp_path / "rec", cfg)
    rec.attach(radar)
    radar.configure(cfg)
    it = radar.frames(timeout=2)
    for _ in range(5):
        next(it)
    radar.stop()
    time.sleep(0.2)
    rec.detach(radar)
    summary = rec.close()
    assert summary["frames"] >= 5
    for name in ("raw_data.bin", "config.cfg", "points.csv", "telemetry.csv",
                 "frames.jsonl", "range_profile.csv"):
        assert (tmp_path / "rec" / name).stat().st_size > 0

    raw = tmp_path / "rec" / "raw_data.bin"
    assert len(list(read_raw_frames(raw))) >= 5
    exp = export_raw(raw, tmp_path / "export")
    assert exp["frames"] >= 5

    replay = ReplaySource(raw, speed=20, loop=False)
    assert replay.params is not None and replay.params.frame_period_ms == 100
    got = []
    replay.add_frame_listener(got.append)
    replay.start()
    deadline = time.time() + 5
    while replay.running and time.time() < deadline:
        time.sleep(0.05)
    replay.close()
    assert len(got) == exp["frames"]
