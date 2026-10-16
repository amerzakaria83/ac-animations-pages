"""Command line interface: ``python -m awr1843 <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_cfg_file, parse_cfg

DEFAULT_CFG = Path(__file__).resolve().parent / "configs" / "awr1843_default.cfg"


def _open_device(args):
    from .device import RadarDevice

    if args.sim:
        return RadarDevice.simulated()
    return RadarDevice.connect(args.cli_port, args.data_port)


def cmd_ports(args) -> int:
    from .device import list_ports

    ports = list_ports()
    if not ports:
        print("No serial ports found. Is the board plugged in and the XDS110 driver "
              "installed (it comes with TI UniFlash / CCS)?")
        return 1
    for p in ports:
        print(p)
    if not any(p.role for p in ports):
        print("\nNo XDS110 ports recognised; pass --cli-port/--data-port explicitly.")
    return 0


def cmd_info(args) -> int:
    params = parse_cfg(load_cfg_file(args.cfg))
    for k, v in params.summary().items():
        print(f"{k:22s} {v}")
    for w in params.warnings:
        print(f"WARNING: {w}")
    return 0


def cmd_send(args) -> int:
    with _open_device(args) as radar:
        ok = True
        for command in args.commands:
            res = radar.send_command(command, timeout=args.timeout)
            print(f"> {command}\n{res.response}")
            ok &= res.ok
    return 0 if ok else 1


def cmd_run(args) -> int:
    from .recorder import Recorder, new_recording_dir

    lines = load_cfg_file(args.cfg)
    with _open_device(args) as radar:
        print(f"Connected: CLI={radar.cli_name} DATA={radar.data_name}")
        ver = radar.version()
        print(ver or "(no version response)")
        recorder = None
        if args.record is not None:
            out = Path(args.record) if args.record else new_recording_dir()
            recorder = Recorder(out, lines)
            recorder.attach(radar)
            print(f"Recording to {out}")

        def progress(i, n, res):
            if not res.ok or args.verbose:
                print(f"  [{i}/{n}] {res}")

        print(f"Sending {args.cfg} ...")
        radar.configure(lines, progress=progress)
        p = radar.params
        print(f"Running: {p.frame_rate_hz:.1f} fps, range res {p.range_resolution_m*100:.1f} cm,"
              f" max range {p.max_range_m:.1f} m, max velocity {p.max_velocity_ms:.2f} m/s")
        t_end = time.time() + args.duration if args.duration else None
        try:
            for frame in radar.frames(timeout=3.0):
                if args.quiet:
                    pass
                elif args.points and len(frame.points):
                    print(f"frame {frame.frame_number}:")
                    for pt in frame.points:
                        print(f"   x={pt['x']:6.2f} y={pt['y']:6.2f} z={pt['z']:6.2f} m "
                              f"v={pt['velocity']:6.2f} m/s  snr={pt['snr_db']:5.1f} dB")
                else:
                    temp = (frame.temperature or {}).get("rx0", float("nan"))
                    print(f"frame {frame.frame_number:6d}  points {len(frame.points):3d}  "
                          f"{radar.stats.fps:5.1f} fps  temp {temp:4.0f} C", end="\r")
                if t_end and time.time() > t_end:
                    break
            else:
                print("\nNo data for 3 s - check the data port / configuration.")
        except KeyboardInterrupt:
            pass
        print()
        radar.stop()
        if recorder:
            recorder.detach(radar)
            s = recorder.close()
            print(f"Saved {s['frames']} frames / {s['points']} points to {s['dir']}")
        st = radar.stats
        print(f"Frames {st.frames}, missed {st.missed_frames}, parse errors {st.parse_errors}")
    return 0


def cmd_export(args) -> int:
    from .recorder import export_raw

    out = args.out or Path(args.raw).with_name("export")
    s = export_raw(args.raw, out, args.cfg)
    print(f"Exported {s['frames']} frames / {s['points']} points to {s['dir']}")
    return 0


def cmd_gui(args) -> int:
    from .gui.app import run_gui

    return run_gui(sim=args.sim, cfg=args.cfg, cli_port=args.cli_port,
                   data_port=args.data_port, replay=args.replay)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="awr1843",
                                 description="Control and read out a TI AWR1843BOOST radar.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command")

    def conn(p):
        p.add_argument("--cli-port", help="CLI/User UART port (e.g. COM3, /dev/ttyACM0)")
        p.add_argument("--data-port", help="Data port (e.g. COM4, /dev/ttyACM1)")
        p.add_argument("--sim", action="store_true", help="use the built-in simulator")

    p = sub.add_parser("gui", help="open the graphical app (default)")
    conn(p)
    p.add_argument("--cfg", default=None, help="configuration file to preselect")
    p.add_argument("--replay", help="raw_data.bin to replay on start")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("ports", help="list serial ports and detect the radar")
    p.set_defaults(func=cmd_ports)

    p = sub.add_parser("info", help="show radar parameters derived from a cfg file")
    p.add_argument("cfg", nargs="?", default=str(DEFAULT_CFG))
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("send", help="send raw CLI command(s) and print the response")
    conn(p)
    p.add_argument("commands", nargs="+", help='e.g. "version" "queryDemoStatus"')
    p.add_argument("--timeout", type=float, default=3.0)
    p.set_defaults(func=cmd_send)

    p = sub.add_parser("run", help="configure, start and stream data in the terminal")
    conn(p)
    p.add_argument("--cfg", default=str(DEFAULT_CFG))
    p.add_argument("--record", nargs="?", const="", default=None,
                   help="record to this folder (default: recordings/<timestamp>)")
    p.add_argument("--duration", type=float, help="stop after N seconds")
    p.add_argument("--points", action="store_true", help="print every detected point")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("export", help="convert a raw_data.bin capture to CSV/NPZ")
    p.add_argument("raw")
    p.add_argument("--cfg", help="config used for the capture (default: config.cfg next to it)")
    p.add_argument("--out", help="output folder")
    p.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if not args.command:
        args = ap.parse_args(["gui"] + (argv if argv is not None else sys.argv[1:]))
    try:
        return args.func(args)
    except Exception as exc:  # friendly message for the common failures
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1
