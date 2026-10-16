"""Desktop GUI (PySide6/PyQt + pyqtgraph)."""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..config import RadarParams, parse_cfg, read_cfg_lines
from ..device import FrameSource, RadarDevice, list_ports
from ..processing import CartesianMapper, angle_axis_deg, range_azimuth_map
from ..recorder import Recorder, ReplaySource, export_raw, new_recording_dir
from ..tlv import Frame, format_platform

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CFG_NAME = "awr1843_default.cfg"
HISTORY = 600

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal


class Bridge(QtCore.QObject):
    """Carries events from worker / IO threads to the GUI thread."""

    log = Signal(str)
    task_done = Signal(str, bool, str)
    progress = Signal(int, int)


def _group(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(8, 6, 8, 8)
    return box, lay


def _row(*widgets) -> QtWidgets.QHBoxLayout:
    lay = QtWidgets.QHBoxLayout()
    for w in widgets:
        if isinstance(w, int):
            lay.addStretch(w)
        else:
            lay.addWidget(w)
    return lay


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sim: bool = False, cfg: str | None = None,
                 cli_port: str | None = None, data_port: str | None = None):
        super().__init__()
        self.setWindowTitle("AWR1843 Radar Studio")
        self.resize(1500, 920)

        self.source: FrameSource | None = None
        self.recorder: Recorder | None = None
        self.params: RadarParams = RadarParams()
        self.busy = False
        self._incoming: deque[Frame] = deque(maxlen=500)
        self._persist: deque[np.ndarray] = deque(maxlen=5)
        self._history: deque[tuple] = deque(maxlen=HISTORY)
        self._t0: float | None = None
        self._last_frame: Frame | None = None
        self._ra_mapper: CartesianMapper | None = None
        self._ra_key = None

        self.bridge = Bridge()
        self.bridge.log.connect(self._append_console)
        self.bridge.task_done.connect(self._on_task_done)
        self.bridge.progress.connect(self._on_progress)

        self._build_ui()
        self._refresh_ports(cli_port, data_port)
        self._load_config_list(cfg)
        if sim:
            self.source_combo.setCurrentIndex(1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)
        self._update_controls()

    # ================================================================== UI
    def _build_ui(self) -> None:
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)

        # ---------------------------------------------------- left panel
        panel = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(panel)
        left.setContentsMargins(6, 6, 6, 6)

        box, lay = _group("1. Connection")
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["AWR1843BOOST (USB)", "Simulator (no hardware)"])
        self.source_combo.currentIndexChanged.connect(self._update_controls)
        lay.addWidget(self.source_combo)
        form = QtWidgets.QFormLayout()
        self.cli_combo = QtWidgets.QComboBox(editable=True)
        self.data_combo = QtWidgets.QComboBox(editable=True)
        form.addRow("CLI port", self.cli_combo)
        form.addRow("Data port", self.data_combo)
        lay.addLayout(form)
        self.refresh_btn = QtWidgets.QPushButton("Detect ports")
        self.refresh_btn.clicked.connect(lambda: self._refresh_ports())
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_connect)
        lay.addLayout(_row(self.refresh_btn, self.connect_btn))
        self.conn_label = QtWidgets.QLabel("Not connected")
        self.conn_label.setWordWrap(True)
        lay.addWidget(self.conn_label)
        left.addWidget(box)

        box, lay = _group("2. Configuration")
        self.cfg_combo = QtWidgets.QComboBox()
        self.cfg_combo.currentIndexChanged.connect(self._on_cfg_selected)
        browse = QtWidgets.QPushButton("...")
        browse.setFixedWidth(32)
        browse.setToolTip("Load a .cfg file")
        browse.clicked.connect(self._browse_cfg)
        lay.addLayout(_row(self.cfg_combo, browse))
        self.params_label = QtWidgets.QLabel()
        self.params_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(self.params_label)
        self.warn_label = QtWidgets.QLabel()
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #e0a030")
        lay.addWidget(self.warn_label)
        left.addWidget(box)

        box, lay = _group("3. Sensor")
        self.start_btn = QtWidgets.QPushButton("Send config && Start")
        self.start_btn.setStyleSheet("font-weight: bold")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.resume_btn = QtWidgets.QPushButton("Resume")
        self.resume_btn.setToolTip("sensorStart 0: restart without re-sending the configuration")
        self.resume_btn.clicked.connect(self._resume)
        lay.addWidget(self.start_btn)
        lay.addLayout(_row(self.stop_btn, self.resume_btn))
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        left.addWidget(box)

        box, lay = _group("4. Live tuning")
        self.clutter_cb = QtWidgets.QCheckBox("Remove static clutter")
        lay.addWidget(self.clutter_cb)
        form = QtWidgets.QFormLayout()
        self.cfar_range = QtWidgets.QDoubleSpinBox(minimum=0, maximum=100, suffix=" dB",
                                                   decimals=1, value=15)
        self.cfar_doppler = QtWidgets.QDoubleSpinBox(minimum=0, maximum=100, suffix=" dB",
                                                     decimals=1, value=15)
        form.addRow("CFAR range threshold", self.cfar_range)
        form.addRow("CFAR Doppler threshold", self.cfar_doppler)
        lay.addLayout(form)
        self.apply_btn = QtWidgets.QPushButton("Apply to radar")
        self.apply_btn.setToolTip("Stops the sensor, sends the changed commands, resumes")
        self.apply_btn.clicked.connect(self._apply_live)
        lay.addWidget(self.apply_btn)
        left.addWidget(box)

        box, lay = _group("5. Record / replay")
        self.rec_dir = QtWidgets.QLineEdit(str(Path.cwd() / "recordings"))
        lay.addWidget(self.rec_dir)
        self.rec_btn = QtWidgets.QPushButton("● Record")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_record)
        self.replay_btn = QtWidgets.QPushButton("Replay...")
        self.replay_btn.clicked.connect(self._open_replay)
        self.export_btn = QtWidgets.QPushButton("Export...")
        self.export_btn.setToolTip("Convert a raw_data.bin capture to CSV / NPZ")
        self.export_btn.clicked.connect(self._export_raw)
        lay.addLayout(_row(self.rec_btn, self.replay_btn, self.export_btn))
        self.rec_label = QtWidgets.QLabel("")
        self.rec_label.setWordWrap(True)
        lay.addWidget(self.rec_label)
        left.addWidget(box)

        box, lay = _group("Status")
        self.status_label = QtWidgets.QLabel()
        self.status_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(self.status_label)
        left.addWidget(box)
        left.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        split.addWidget(scroll)

        # ---------------------------------------------------------- tabs
        self.tabs = QtWidgets.QTabWidget()
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.tabs.addTab(self._build_points_tab(), "Point cloud")
        self.tabs.addTab(self._build_heatmap_tab(), "Heatmaps")
        self.tabs.addTab(self._build_telemetry_tab(), "Telemetry")
        self.tabs.addTab(self._build_table_tab(), "Points table")
        self.tabs.addTab(self._build_console_tab(), "CLI console")
        self.tabs.addTab(self._build_cfg_tab(), "Config editor")
        self.statusBar().showMessage("Connect the radar (or pick the simulator) to begin.")

    def _build_points_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.color_by = QtWidgets.QComboBox()
        self.color_by.addItems(["Velocity", "SNR", "Height (z)"])
        self.persist_spin = QtWidgets.QSpinBox(minimum=1, maximum=50, value=5)
        self.persist_spin.valueChanged.connect(
            lambda n: setattr(self, "_persist", deque(self._persist, maxlen=n)))
        self.min_snr = QtWidgets.QDoubleSpinBox(minimum=0, maximum=60, value=0, suffix=" dB")
        self.hide_static = QtWidgets.QCheckBox("Hide static points")
        self.color_legend = QtWidgets.QLabel()
        v.addLayout(_row(QtWidgets.QLabel("Color by"), self.color_by,
                         QtWidgets.QLabel("  Persistence (frames)"), self.persist_spin,
                         QtWidgets.QLabel("  Min SNR"), self.min_snr, self.hide_static,
                         self.color_legend, 1))

        glw = pg.GraphicsLayoutWidget()
        v.addWidget(glw, 1)
        self.top_plot = glw.addPlot(row=0, col=0, rowspan=2, title="Top view")
        self.top_plot.setLabel("bottom", "x (lateral)", "m")
        self.top_plot.setLabel("left", "y (forward)", "m")
        self.top_plot.setAspectLocked(True)
        self.top_plot.showGrid(x=True, y=True, alpha=0.2)
        self.top_scatter = pg.ScatterPlotItem(size=8, pen=None)
        self.top_plot.addItem(self.top_scatter)
        self._fov_items: list = []

        self.side_plot = glw.addPlot(row=0, col=1, title="Side view")
        self.side_plot.setLabel("bottom", "y (forward)", "m")
        self.side_plot.setLabel("left", "z (height)", "m")
        self.side_plot.showGrid(x=True, y=True, alpha=0.2)
        self.side_scatter = pg.ScatterPlotItem(size=7, pen=None)
        self.side_plot.addItem(self.side_scatter)

        self.rv_plot = glw.addPlot(row=1, col=1, title="Range / velocity")
        self.rv_plot.setLabel("bottom", "range", "m")
        self.rv_plot.setLabel("left", "radial velocity", "m/s")
        self.rv_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rv_scatter = pg.ScatterPlotItem(size=7, pen=None)
        self.rv_plot.addItem(self.rv_scatter)

        self.rp_plot = glw.addPlot(row=2, col=0, colspan=2, title="Range profile")
        self.rp_plot.setLabel("bottom", "range", "m")
        self.rp_plot.setLabel("left", "relative power", "dB")
        self.rp_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rp_plot.addLegend(offset=(-10, 10), brush=pg.mkBrush(0, 0, 0, 170))
        self.rp_curve = self.rp_plot.plot(pen=pg.mkPen("#4fc3f7", width=1.5), name="signal")
        self.np_curve = self.rp_plot.plot(pen=pg.mkPen("#ff8a65", width=1), name="noise")
        self.rp_points = pg.ScatterPlotItem(size=7, brush=pg.mkBrush("#ffee58"), pen=None,
                                            symbol="t", name="detections")
        self.rp_plot.addItem(self.rp_points)
        glw.ci.layout.setRowStretchFactor(0, 3)
        glw.ci.layout.setRowStretchFactor(1, 3)
        glw.ci.layout.setRowStretchFactor(2, 2)
        glw.ci.layout.setColumnStretchFactor(0, 3)
        glw.ci.layout.setColumnStretchFactor(1, 2)
        return w

    def _build_heatmap_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.heat_hint = QtWidgets.QLabel(
            "Heatmaps are only sent when enabled in <code>guiMonitor</code>. Select "
            "<b>awr1843_heatmaps.cfg</b> (the UART is too slow to send them at 10 fps).")
        self.heat_hint.setWordWrap(True)
        self.mirror_cb = QtWidgets.QCheckBox("Mirror azimuth")
        self.ra_db_cb = QtWidgets.QCheckBox("Log scale (dB)")
        self.ra_db_cb.setChecked(True)
        for cb in (self.ra_db_cb, self.mirror_cb):
            cb.toggled.connect(lambda _: self.ra_img.clear())  # re-fit color levels
        v.addLayout(_row(self.heat_hint, 1, self.ra_db_cb, self.mirror_cb))
        glw = pg.GraphicsLayoutWidget()
        v.addWidget(glw, 1)
        cmap = pg.colormap.get("viridis")

        self.rd_plot = glw.addPlot(row=0, col=0, title="Range-Doppler")
        self.rd_plot.setLabel("bottom", "radial velocity", "m/s")
        self.rd_plot.setLabel("left", "range", "m")
        self.rd_img = pg.ImageItem()
        self.rd_plot.addItem(self.rd_img)
        self.rd_bar = pg.ColorBarItem(colorMap=cmap, label="dB", interactive=True)
        self.rd_bar.setImageItem(self.rd_img, insert_in=self.rd_plot)

        self.ra_plot = glw.addPlot(row=0, col=1, title="Range-Azimuth (static, zero Doppler)")
        self.ra_plot.setLabel("bottom", "x (lateral)", "m")
        self.ra_plot.setLabel("left", "y (forward)", "m")
        self.ra_plot.setAspectLocked(True)
        self.ra_img = pg.ImageItem()
        self.ra_plot.addItem(self.ra_img)
        self.ra_bar = pg.ColorBarItem(colorMap=cmap, interactive=True)
        self.ra_bar.setImageItem(self.ra_img, insert_in=self.ra_plot)
        self.ra_overlay = pg.ScatterPlotItem(size=6, pen=pg.mkPen("w"), brush=None)
        self.ra_plot.addItem(self.ra_overlay)
        return w

    def _build_telemetry_tab(self) -> QtWidgets.QWidget:
        glw = pg.GraphicsLayoutWidget()
        self.tele_curves = {}

        def plot(row, title, ylabel, series):
            p = glw.addPlot(row=row, col=0, title=title)
            p.setLabel("left", ylabel)
            p.setLabel("bottom", "time", "s")
            p.showGrid(x=True, y=True, alpha=0.2)
            p.addLegend(offset=(10, 5), brush=pg.mkBrush(0, 0, 0, 170))
            for key, name, color in series:
                self.tele_curves[key] = p.plot(pen=pg.mkPen(color, width=1.5), name=name)
            return p

        p1 = plot(0, "Detections", "points", [("points", "points / frame", "#4fc3f7")])
        p2 = plot(1, "DSP / CPU", "%", [("active_load", "active frame CPU load", "#81c784"),
                                        ("inter_load", "inter-frame CPU load", "#ffb74d")])
        p3 = plot(2, "Timing", "ms", [("margin", "inter-frame margin", "#ba68c8"),
                                      ("tx_time", "UART transmit time", "#e57373"),
                                      ("proc", "inter-frame processing", "#4db6ac")])
        p4 = plot(3, "Temperature", "°C", [("t_rx0", "RX0", "#4fc3f7"),
                                                ("t_tx0", "TX0", "#e57373"),
                                                ("t_pm", "PM", "#fff176"),
                                                ("t_dig0", "DIG0", "#aed581")])
        for p in (p2, p3, p4):
            p.setXLink(p1)
        return glw

    def _build_table_tab(self) -> QtWidgets.QWidget:
        self.table_cols = ["x", "y", "z", "velocity", "range", "azimuth_deg",
                           "elevation_deg", "snr_db", "noise_db"]
        self.table = QtWidgets.QTableWidget(0, len(self.table_cols))
        self.table.setHorizontalHeaderLabels(
            ["x (m)", "y (m)", "z (m)", "v (m/s)", "range (m)", "azimuth (°)",
             "elevation (°)", "SNR (dB)", "noise (dB)"])
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.table_label = QtWidgets.QLabel("Detected points of the latest frame")
        v.addWidget(self.table_label)
        v.addWidget(self.table)
        return w

    def _build_console_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.console = QtWidgets.QPlainTextEdit(readOnly=True)
        self.console.setMaximumBlockCount(5000)
        self.console.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        v.addWidget(self.console)
        self.cmd_edit = QtWidgets.QLineEdit()
        self.cmd_edit.setPlaceholderText("CLI command, e.g. version, queryDemoStatus, help, "
                                         "clutterRemoval -1 1 ...")
        self.cmd_edit.returnPressed.connect(self._send_console_cmd)
        send = QtWidgets.QPushButton("Send")
        send.clicked.connect(self._send_console_cmd)
        quick = []
        for cmd in ("version", "queryDemoStatus", "help"):
            b = QtWidgets.QPushButton(cmd)
            b.clicked.connect(lambda _=False, c=cmd: self._send_command(c))
            quick.append(b)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.console.clear)
        v.addLayout(_row(self.cmd_edit, send, *quick, clear))
        return w

    def _build_cfg_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            "This text is exactly what <b>Send config &amp; Start</b> sends to the radar. "
            "Edit freely; derived parameters update on the left."))
        self.cfg_edit = QtWidgets.QPlainTextEdit()
        self.cfg_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.cfg_edit.textChanged.connect(lambda: self._cfg_debounce.start(300))
        self._cfg_debounce = QtCore.QTimer(self, singleShot=True)
        self._cfg_debounce.timeout.connect(self._update_params_from_editor)
        v.addWidget(self.cfg_edit)
        save = QtWidgets.QPushButton("Save as...")
        save.clicked.connect(self._save_cfg)
        revert = QtWidgets.QPushButton("Reload from file")
        revert.clicked.connect(self._on_cfg_selected)
        v.addLayout(_row(1, revert, save))
        return w

    # ============================================================ helpers
    def _log(self, text: str) -> None:
        self.bridge.log.emit(text)

    def _append_console(self, text: str) -> None:
        self.console.appendPlainText(text.rstrip())

    def _run_task(self, name: str, fn) -> None:
        """Run ``fn`` in a worker thread; result arrives in ``_on_task_done``."""
        if self.busy:
            return
        self.busy = True
        self._update_controls()

        def work():
            try:
                msg = fn() or ""
                self.bridge.task_done.emit(name, True, str(msg))
            except Exception as exc:
                self._log(traceback.format_exc())
                self.bridge.task_done.emit(name, False, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_task_done(self, name: str, ok: bool, msg: str) -> None:
        self.busy = False
        self.progress.setVisible(False)
        if ok:
            self.statusBar().showMessage(f"{name}: done. {msg}", 8000)
        else:
            self.statusBar().showMessage(f"{name} failed: {msg}")
            QtWidgets.QMessageBox.warning(self, f"{name} failed", msg)
        if name == "Connect":
            if ok:
                self.conn_label.setText(msg)
            else:
                self._disconnect()
        self._update_controls()

    def _on_progress(self, i: int, n: int) -> None:
        self.progress.setVisible(True)
        self.progress.setMaximum(n)
        self.progress.setValue(i)

    @property
    def device(self) -> RadarDevice | None:
        return self.source if isinstance(self.source, RadarDevice) else None

    def _update_controls(self) -> None:
        connected = self.source is not None
        dev = self.device
        replay = isinstance(self.source, ReplaySource)
        idle = not self.busy
        usb = self.source_combo.currentIndex() == 0
        self.source_combo.setEnabled(not connected)
        self.cli_combo.setEnabled(not connected and usb)
        self.data_combo.setEnabled(not connected and usb)
        self.refresh_btn.setEnabled(not connected and usb)
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.connect_btn.setEnabled(idle)
        running = bool(dev and dev.running) or bool(replay and not self.source.paused)
        self.start_btn.setEnabled(idle and dev is not None)
        self.stop_btn.setEnabled(idle and connected and running)
        self.resume_btn.setEnabled(idle and connected and not running
                                   and (replay or bool(dev and dev.cfg_lines)))
        self.apply_btn.setEnabled(idle and dev is not None and bool(dev.cfg_lines))
        self.rec_btn.setEnabled(dev is not None or self.recorder is not None)
        self.replay_btn.setEnabled(not connected or replay)

    # ======================================================== connection
    def _refresh_ports(self, cli: str | None = None, data: str | None = None) -> None:
        ports = list_ports()
        for combo, role, explicit in ((self.cli_combo, "cli", cli),
                                      (self.data_combo, "data", data)):
            combo.clear()
            for p in ports:
                combo.addItem(str(p), p.device)
            match = [i for i, p in enumerate(ports) if p.role == role]
            if explicit:
                combo.setEditText(explicit)
            elif match:
                combo.setCurrentIndex(match[0])
        found = sum(1 for p in ports if p.role)
        self.statusBar().showMessage(
            f"Found {len(ports)} serial ports"
            + (f", radar detected on {found} of them." if found else
               ". Radar not auto-detected - choose the ports manually."), 6000)

    @staticmethod
    def _combo_port(combo: QtWidgets.QComboBox) -> str:
        idx = combo.currentIndex()
        if idx >= 0 and combo.currentText() == combo.itemText(idx):
            return combo.itemData(idx)
        return combo.currentText().split(" - ")[0].strip()

    def _toggle_connect(self) -> None:
        if self.source is not None:
            self._disconnect()
            return
        sim = self.source_combo.currentIndex() == 1
        cli, data = self._combo_port(self.cli_combo), self._combo_port(self.data_combo)
        try:
            dev = RadarDevice.simulated() if sim else RadarDevice.connect(cli, data)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Connection failed",
                f"{exc}\n\nCheck that the board is plugged in, powered by its 5 V supply, "
                "flashed with the mmWave SDK out-of-box demo and in functional mode "
                "(SOP jumpers), and that no other program (e.g. the TI visualizer) "
                "has the ports open.")
            return
        self._attach_source(dev)
        dev.cli_listeners.append(self._log)
        self._log(f"--- connected: CLI={dev.cli_name} DATA={dev.data_name}")

        def check():
            res = dev.send_command("version")
            if not res.ok:
                raise RuntimeError(res.response)
            info = [ln.split(":", 1)[1].strip() for ln in res.response.splitlines()
                    if ":" in ln and ("Platform" in ln or "SDK" in ln)]
            return "Connected - " + (", ".join(info) or "firmware responded")

        self._run_task("Connect", check)

    def _attach_source(self, source: FrameSource) -> None:
        self.source = source
        source.add_frame_listener(self._incoming.append)
        self._reset_views()

    def _disconnect(self) -> None:
        if self.rec_btn.isChecked():
            self.rec_btn.setChecked(False)
        src, self.source = self.source, None
        if src is not None:
            src.remove_frame_listener(self._incoming.append)
            self._log("--- disconnected")
            threading.Thread(target=src.close, daemon=True).start()
        self.conn_label.setText("Not connected")
        self._update_controls()

    # ====================================================== configuration
    def _load_config_list(self, select: str | None = None) -> None:
        self.cfg_combo.blockSignals(True)
        self.cfg_combo.clear()
        for path in sorted(CONFIG_DIR.glob("*.cfg")):
            self.cfg_combo.addItem(path.name, str(path))
        self.cfg_combo.blockSignals(False)
        if select:
            self._add_cfg(select)
        else:
            idx = self.cfg_combo.findText(DEFAULT_CFG_NAME)
            self.cfg_combo.setCurrentIndex(max(idx, 0))
            self._on_cfg_selected()

    def _add_cfg(self, path: str) -> None:
        idx = self.cfg_combo.findData(str(Path(path).resolve()))
        if idx < 0:
            self.cfg_combo.addItem(Path(path).name, str(Path(path).resolve()))
            idx = self.cfg_combo.count() - 1
        self.cfg_combo.setCurrentIndex(idx)
        self._on_cfg_selected()

    def _browse_cfg(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open configuration", str(CONFIG_DIR), "mmWave config (*.cfg);;All files (*)")
        if path:
            self._add_cfg(path)

    def _on_cfg_selected(self, *_) -> None:
        path = self.cfg_combo.currentData()
        if not path:
            return
        try:
            self.cfg_edit.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot read config", str(exc))
        self._update_params_from_editor()

    def _save_cfg(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save configuration", str(CONFIG_DIR / "my_config.cfg"), "*.cfg")
        if path:
            Path(path).write_text(self.cfg_edit.toPlainText())
            self._add_cfg(path)

    def _cfg_lines(self) -> list[str]:
        return read_cfg_lines(self.cfg_edit.toPlainText())

    def _update_params_from_editor(self) -> None:
        params = parse_cfg(self._cfg_lines())
        rows = "".join(f"<tr><td>{k}</td><td>&nbsp;<b>{v}</b></td></tr>"
                       for k, v in params.summary().items())
        self.params_label.setText(f"<table>{rows}</table>")
        self.warn_label.setText("\n".join(f"⚠ {w}" for w in params.warnings))
        self.clutter_cb.setChecked(params.clutter_removal)
        if params.cfar_range_threshold_db is not None:
            self.cfar_range.setValue(params.cfar_range_threshold_db)
        if params.cfar_doppler_threshold_db is not None:
            self.cfar_doppler.setValue(params.cfar_doppler_threshold_db)
        if self.source is None or (self.device and not self.device.running):
            self._set_params(params)

    def _set_params(self, params: RadarParams) -> None:
        self.params = params
        self._ra_mapper = None
        r = params.max_range_m
        for item in self._fov_items:
            self.top_plot.removeItem(item)
        self._fov_items = []
        pen = pg.mkPen((120, 120, 120), width=1, style=QtCore.Qt.PenStyle.DashLine)
        for ang in (-60, 60):
            a = np.radians(ang)
            line = pg.PlotCurveItem([0, r * np.sin(a)], [0, r * np.cos(a)], pen=pen)
            self.top_plot.addItem(line)
            self._fov_items.append(line)
        th = np.linspace(-np.pi / 2, np.pi / 2, 90)
        step = 1 if r <= 12 else 5
        for ring in np.arange(step, r + 1e-6, step):
            c = pg.PlotCurveItem(ring * np.sin(th), ring * np.cos(th),
                                 pen=pg.mkPen((70, 70, 70), width=1))
            self.top_plot.addItem(c)
            self._fov_items.append(c)
        self.top_plot.setXRange(-r * 0.8, r * 0.8)
        self.top_plot.setYRange(0, r)
        self.side_plot.setXRange(0, r)
        self.side_plot.setYRange(-2, 2)
        self.rv_plot.setXRange(0, r)
        v = params.max_velocity_ms
        self.rv_plot.setYRange(-v, v)
        self.rp_plot.setXRange(0, params.num_range_bins * params.range_resolution_m)

    # ===================================================== sensor control
    def _start(self) -> None:
        dev = self.device
        if dev is None:
            return
        lines = self._cfg_lines()
        params = parse_cfg(lines)
        self._set_params(params)
        self._reset_views()

        def go():
            dev.configure(lines, progress=lambda i, n, r: self.bridge.progress.emit(i, n))
            return f"{params.frame_rate_hz:.1f} fps"

        self._run_task("Start", go)

    def _stop(self) -> None:
        if isinstance(self.source, ReplaySource):
            self.source.paused = True
            self._update_controls()
        elif self.device:
            self._run_task("Stop", lambda: self.device.stop().response)

    def _resume(self) -> None:
        if isinstance(self.source, ReplaySource):
            self.source.paused = False
            self._update_controls()
        elif self.device:
            def go():
                res = self.device.start()
                if not res.ok:
                    raise RuntimeError(res.response)
            self._run_task("Resume", go)

    def _replace_cfg_line(self, prefix: list[str], new_line: str) -> None:
        lines = self.cfg_edit.toPlainText().splitlines()
        for i, line in enumerate(lines):
            if line.split()[:len(prefix)] == prefix:
                lines[i] = new_line
                break
        else:
            lines.insert(max(0, len(lines) - 1), new_line)
        self.cfg_edit.blockSignals(True)
        self.cfg_edit.setPlainText("\n".join(lines) + "\n")
        self.cfg_edit.blockSignals(False)

    def _apply_live(self) -> None:
        dev = self.device
        if dev is None:
            return
        cmds = [f"clutterRemoval -1 {int(self.clutter_cb.isChecked())}"]
        self._replace_cfg_line(["clutterRemoval"], cmds[0])
        defaults = {"0": "cfarCfg -1 0 2 8 4 3 0 15 1", "1": "cfarCfg -1 1 0 4 2 3 1 15 1"}
        for direction, spin in (("0", self.cfar_range), ("1", self.cfar_doppler)):
            base = next((ln for ln in dev.cfg_lines
                         if ln.split()[:3] == ["cfarCfg", "-1", direction]),
                        defaults[direction]).split()
            base[8] = f"{spin.value():g}"
            cmds.append(" ".join(base))
            self._replace_cfg_line(["cfarCfg", "-1", direction], cmds[-1])

        def go():
            results = dev.apply_live(cmds)
            bad = [str(r) for r in results if not r.ok]
            if bad:
                raise RuntimeError("\n".join(bad))
            return f"sent {len(cmds)} commands"

        self._run_task("Apply", go)

    def _send_console_cmd(self) -> None:
        cmd = self.cmd_edit.text().strip()
        if cmd:
            self.cmd_edit.clear()
            self._send_command(cmd)

    def _send_command(self, cmd: str) -> None:
        dev = self.device
        if dev is None:
            self._append_console("(not connected)")
            return

        def go():
            res = dev.send_command(cmd, timeout=5.0)
            if cmd.startswith("sensorStop"):
                dev.running = False
            elif cmd.startswith("sensorStart") and res.ok:
                dev.running = True
            return "OK" if res.ok else "returned an error"

        self._run_task(f"'{cmd}'", go)

    # ============================================================ record
    def _toggle_record(self, on: bool) -> None:
        if on:
            if self.device is None:
                self.rec_btn.setChecked(False)
                return
            out = new_recording_dir(self.rec_dir.text() or "recordings")
            self.recorder = Recorder(out, self.device.cfg_lines or self._cfg_lines())
            self.recorder.attach(self.device)
            self.rec_btn.setText("■ Stop recording")
            self.rec_btn.setStyleSheet("color: #ff5252; font-weight: bold")
        elif self.recorder is not None:
            rec, self.recorder = self.recorder, None
            if self.source is not None:
                rec.detach(self.source)
            s = rec.close()
            self.rec_btn.setText("● Record")
            self.rec_btn.setStyleSheet("")
            self.rec_label.setText(f"Saved {s['frames']} frames, {s['points']} points to "
                                   f"{s['dir']}")
        self._update_controls()

    def _open_replay(self) -> None:
        if isinstance(self.source, ReplaySource):
            self._disconnect()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open recording", self.rec_dir.text(), "Raw capture (raw_data.bin *.bin)")
        if path:
            self.start_replay(path)

    def start_replay(self, path: str) -> None:
        src = ReplaySource(path)
        if src.params is not None:
            self._set_params(src.params)
            if src.cfg_lines:
                self.cfg_edit.blockSignals(True)
                self.cfg_edit.setPlainText("\n".join(src.cfg_lines))
                self.cfg_edit.blockSignals(False)
        self._attach_source(src)
        src.start()
        self.conn_label.setText(f"Replaying {path}")
        self._update_controls()

    def _export_raw(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Raw capture to export", self.rec_dir.text(), "Raw capture (*.bin)")
        if not path:
            return
        out = str(Path(path).with_name("export"))
        self._run_task("Export", lambda: "{frames} frames -> {dir}".format(
            **export_raw(path, out)))

    # =========================================================== drawing
    def _reset_views(self) -> None:
        self._incoming.clear()
        self._persist.clear()
        self._history.clear()
        self._t0 = None
        self._last_frame = None

    def _tick(self) -> None:
        frames = []
        while self._incoming:
            frames.append(self._incoming.popleft())
        if self.source is not None and self.source.params is not None \
                and self.source.params is not self.params \
                and isinstance(self.source, RadarDevice) and self.source.running:
            self._set_params(self.source.params)
        for f in frames:
            if self._t0 is None:
                self._t0 = f.host_time
            self._persist.append(f.points)
            s, t = f.stats or {}, f.temperature or {}
            self._history.append((
                f.host_time - self._t0, len(f.points),
                s.get("active_frame_cpu_load_pct", np.nan),
                s.get("inter_frame_cpu_load_pct", np.nan),
                s.get("inter_frame_processing_margin_us", np.nan) / 1000,
                s.get("transmit_output_time_us", np.nan) / 1000,
                s.get("inter_frame_processing_time_us", np.nan) / 1000,
                t.get("rx0", np.nan), t.get("tx0", np.nan), t.get("pm", np.nan),
                t.get("dig0", np.nan)))
        if frames:
            self._last_frame = frames[-1]
            self._draw_frame(frames[-1])
            self._draw_telemetry()
        self._draw_status()
        self._update_controls()  # cheap; picks up state changed from the console

    def _filtered_points(self) -> np.ndarray:
        pts = np.concatenate(list(self._persist)) if self._persist else np.zeros(0)
        if len(pts) == 0:
            return pts
        if self.min_snr.value() > 0:
            pts = pts[~(pts["snr_db"] < self.min_snr.value())]
        if self.hide_static.isChecked():
            pts = pts[np.abs(pts["velocity"]) > 1e-3]
        return pts

    def _brushes(self, pts: np.ndarray):
        mode = self.color_by.currentIndex()
        if mode == 0:
            v = self.params.max_velocity_ms or 1
            vals, cmap, lo, hi = pts["velocity"], "CET-D1", -v, v
            label = f"blue = approaching, red = receding (±{v:.1f} m/s)"
        elif mode == 1:
            snr = np.nan_to_num(pts["snr_db"], nan=0)
            vals, cmap, lo, hi = snr, "viridis", 5, 35
            label = "SNR 5 → 35 dB"
        else:
            vals, cmap, lo, hi = pts["z"], "plasma", -1.5, 1.5
            label = "z -1.5 → 1.5 m"
        self.color_legend.setText(f"  {label}")
        norm = np.clip((np.asarray(vals, float) - lo) / (hi - lo), 0, 1)
        colors = pg.colormap.get(cmap).map(norm, mode="byte")
        return [pg.mkBrush(*c) for c in colors]

    def _draw_frame(self, f: Frame) -> None:
        p = self.params
        pts = self._filtered_points()
        if len(pts):
            brushes = self._brushes(pts)
            self.top_scatter.setData(pts["x"], pts["y"], brush=brushes)
            self.side_scatter.setData(pts["y"], pts["z"], brush=brushes)
            self.rv_scatter.setData(pts["range"], pts["velocity"], brush=brushes)
        else:
            for s in (self.top_scatter, self.side_scatter, self.rv_scatter):
                s.setData([], [])

        rp = f.range_profile_db
        if rp is not None:
            axis = np.arange(len(rp)) * p.range_resolution_m
            self.rp_curve.setData(axis, rp)
            if len(f.points):
                idx = np.clip(np.round(f.points["range"] / p.range_resolution_m).astype(int),
                              0, len(rp) - 1)
                self.rp_points.setData(f.points["range"], rp[idx] + 3)
            else:
                self.rp_points.setData([], [])
        npf = f.noise_profile_db
        if npf is not None:
            self.np_curve.setData(np.arange(len(npf)) * p.range_resolution_m, npf)

        if self.tabs.currentIndex() == 1:
            self._draw_heatmaps(f)
        if self.tabs.currentIndex() == 3:
            self._draw_table(f)

    def _draw_heatmaps(self, f: Frame) -> None:
        p = self.params
        have = False
        rd = f.range_doppler_db
        if rd is not None:
            have = True
            n_r, n_d = rd.shape
            first = self.rd_img.image is None
            self.rd_img.setImage(rd, autoLevels=False)
            v = p.velocity_resolution_ms
            self.rd_img.setRect(QtCore.QRectF(-(n_d // 2 + 0.5) * v, -0.5 * p.range_resolution_m,
                                              n_d * v, n_r * p.range_resolution_m))
            if first:
                lo, hi = np.percentile(rd, [5, 99.9])
                self.rd_bar.setLevels((lo, hi))
        if f.azimuth_static is not None:
            have = True
            ra = range_azimuth_map(f.azimuth_static, 64, mirror=self.mirror_cb.isChecked())
            if self.ra_db_cb.isChecked():
                ra = 20 * np.log10(ra + 1)
            key = (ra.shape, p.range_resolution_m)
            if self._ra_mapper is None or self._ra_key != key:
                self._ra_mapper = CartesianMapper(ra.shape[0], p.range_resolution_m, ra.shape[1],
                                                  grid=240, max_range=min(p.max_range_m,
                                                                           ra.shape[0] * p.range_resolution_m))
                self._ra_key = key
                self.ra_img.clear()
                x0, _, w, h = self._ra_mapper.extent
                self.ra_plot.setXRange(x0, x0 + w, padding=0.02)
                self.ra_plot.setYRange(0, h, padding=0.02)
            img = self._ra_mapper(ra).T
            first = self.ra_img.image is None
            self.ra_img.setImage(img, autoLevels=False)
            x0, y0, w, h = self._ra_mapper.extent
            self.ra_img.setRect(QtCore.QRectF(x0, y0, w, h))
            if first:
                valid = img[self._ra_mapper.valid.T]
                lo, hi = np.percentile(valid, [20, 99.9]) if valid.size else (0, 1)
                self.ra_bar.setLevels((lo, hi))
            pts = f.points
            self.ra_overlay.setData(pts["x"], pts["y"]) if len(pts) else self.ra_overlay.setData([], [])
        self.heat_hint.setVisible(not have)

    def _draw_table(self, f: Frame) -> None:
        pts = f.points
        self.table_label.setText(f"Frame {f.frame_number}: {len(pts)} detected points")
        self.table.setRowCount(len(pts))
        for r, pt in enumerate(pts):
            for c, col in enumerate(self.table_cols):
                val = float(pt[col])
                item = QtWidgets.QTableWidgetItem("" if np.isnan(val) else f"{val:.3f}")
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                      | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)

    def _draw_telemetry(self) -> None:
        if not self._history:
            return
        h = np.array(self._history, dtype=float)
        t = h[:, 0]
        for key, col in (("points", 1), ("active_load", 2), ("inter_load", 3), ("margin", 4),
                         ("tx_time", 5), ("proc", 6), ("t_rx0", 7), ("t_tx0", 8),
                         ("t_pm", 9), ("t_dig0", 10)):
            y = h[:, col]
            if np.all(np.isnan(y)):
                self.tele_curves[key].setData([], [])
            else:
                self.tele_curves[key].setData(t, y, connect="finite")

    def _draw_status(self) -> None:
        src = self.source
        rows = []
        if src is not None:
            st = src.stats
            f = self._last_frame
            rows += [("Frames", f"{st.frames}"), ("Frame rate", f"{st.fps:.1f} fps"),
                     ("Data rate", f"{st.bytes_per_sec / 1024:.1f} kB/s")]
            if f is not None:
                rows += [("Frame #", f"{f.frame_number}"),
                         ("Points", f"{len(f.points)}"),
                         ("Firmware", f"SDK {f.version_str} / {format_platform(f.platform)}")]
                if f.stats:
                    rows.append(("CPU load", f"{f.stats.get('active_frame_cpu_load_pct', 0)} % "
                                             f"active, {f.stats.get('inter_frame_cpu_load_pct', 0)} %"
                                             " inter"))
                if f.temperature:
                    temps = [v for k, v in f.temperature.items() if k not in ("valid", "time_ms")]
                    rows.append(("Temperature", f"{min(temps):.0f}-{max(temps):.0f} °C"))
            rows += [("Missed frames", f"{st.missed_frames}"),
                     ("Parse errors", f"{st.parse_errors}"),
                     ("Discarded bytes", f"{st.bytes_discarded}")]
            if isinstance(src, RadarDevice):
                rows.insert(0, ("Sensor", "<span style='color:#66bb6a'>running</span>"
                                if src.running else "stopped"))
            elif isinstance(src, ReplaySource):
                rows.insert(0, ("Replay", "paused" if src.paused else "playing"))
        if self.recorder:
            r = self.recorder
            self.rec_label.setText(f"Recording to {r.out_dir}\n{r.frames} frames, "
                                   f"{r.points} points, {r.bytes / 1024:.0f} kB")
        html = "".join(f"<tr><td>{k}</td><td>&nbsp;<b>{v}</b></td></tr>" for k, v in rows)
        self.status_label.setText(f"<table>{html}</table>" if rows else "Idle")

    def closeEvent(self, ev) -> None:  # noqa: N802 (Qt API)
        self.timer.stop()
        if self.recorder is not None:
            self.rec_btn.setChecked(False)
        if self.source is not None:
            self.source.close()
        super().closeEvent(ev)


def run_gui(sim: bool = False, cfg: str | None = None, cli_port: str | None = None,
            data_port: str | None = None, replay: str | None = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AWR1843 Radar Studio")
    app.setStyle("Fusion")
    win = MainWindow(sim=sim, cfg=cfg, cli_port=cli_port, data_port=data_port)
    win.show()
    if replay:
        win.start_replay(replay)
    return app.exec()
