# src/tool.py
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDoubleSpinBox,
    QSpinBox, QPushButton, QCheckBox, QGroupBox, QFileDialog, QComboBox
)

from Qt.QtWidgets import QWidget, QToolButton, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy
from Qt.QtCore import Qt


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None, expanded: bool = False):
        super().__init__(parent)

        self.toggle = QToolButton(self)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.clicked.connect(self._on_toggle)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.toggle)
        header.addStretch(1)

        self.content = QFrame(self)
        self.content.setFrameShape(QFrame.NoFrame)
        self.content.setVisible(expanded)

        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 4, 0, 0)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(header)
        outer.addWidget(self.content)

    def _on_toggle(self):
        expanded = self.toggle.isChecked()
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)


class DAQTool(ToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False  # 
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "DAQplugin"
        
        self.tool_window = MainToolWindow(self, close_destroys=True)

        self._build_ui()
        
        self.tool_window.manage(None)

    # ---------------- UI helpers ----------------
    def _browse_open_file(self, line_edit, title="Select file"):
        path, _ = QFileDialog.getOpenFileName(self.tool_window.ui_area, title, "")
        if path:
            line_edit.setText(path)

    def _browse_save_file(self, line_edit, title="Save file"):
        path, _ = QFileDialog.getSaveFileName(self.tool_window.ui_area, title, "")
        if path:
            line_edit.setText(path)

    def _refresh_models(self):
        """Populate structure and volume combos from session models."""
        self.structure_combo.clear()
        self.volume_combo.clear()

        # Import types
        try:
            from chimerax.atomic import Structure
        except Exception:
            Structure = None
        try:
            from chimerax.map import Volume
        except Exception:
            Volume = None

        for m in self.session.models.list():
            if Structure is not None and isinstance(m, Structure):
                self.structure_combo.addItem(f"#{m.id_string} {m.name}", m)
            if Volume is not None and isinstance(m, Volume):
                self.volume_combo.addItem(f"#{m.id_string} {m.name}", m)

    def _selected_structure(self):
        return self.structure_combo.currentData()

    def _selected_volume(self):
        return self.volume_combo.currentData()

    def _map_input_token(self):
        """
        map_input is Or(MapArg, OpenFileNameArg).
        Prefer loaded volume if selected; otherwise use file path.
        """
        vol = self._selected_volume()
        path = self.map_path_edit.text().strip()

        if vol is not None:
            return f"#{vol.id_string}"
        if path:
            # quote to survive spaces
            return f"\"{path}\""
        return None

    def _structure_token_or_none(self):
        st = self._selected_structure()
        if st is None:
            return None
        return f"#{st.id_string}"

    def _optional_kw(self, key, value, quote=False):
        """Return ' key value' or '' if value is empty/None."""
        if value is None:
            return ""
        if isinstance(value, str):
            v = value.strip()
            if v == "":
                return ""
            return f" {key} " + (f"\"{v}\"" if quote else v)
        return f" {key} {value}"

    # ---------------- Build UI ----------------
    def _build_ui(self):
        parent = self.tool_window.ui_area
        root = QWidget(parent)
        main = QVBoxLayout(root)

        # ---- Model / Map selection ----
        box_sel = QGroupBox("Inputs (Loaded models or files)", root)
        lay_sel = QVBoxLayout(box_sel)

        row = QHBoxLayout()
        row.addWidget(QLabel("Structure:", root))
        self.structure_combo = QComboBox(root)
        row.addWidget(self.structure_combo, 1)

        row.addWidget(QLabel("Loaded map:", root))
        self.volume_combo = QComboBox(root)
        row.addWidget(self.volume_combo, 1)




        lay_sel.addLayout(row)



        btn_refresh = QPushButton("Refresh model list", root)
        btn_refresh.clicked.connect(self._refresh_models)
        lay_sel.addWidget(btn_refresh)

        main.addWidget(box_sel)

        # ---- Common options ----
        box_opt = QGroupBox("Common options", root)
        lay_opt = QVBoxLayout(box_opt)

        row = QHBoxLayout()
        row.addWidget(QLabel("metric:", root))
        self.metric_combo = QComboBox(root)
        self.metric_combo.addItems([
            "aa_score",    # DAQ(AA)
            "atom_score",  # DAQ(CA)
        ])
        self.metric_combo.setCurrentText("aa_score")

        row.addWidget(self.metric_combo, 1)
        lay_opt.addLayout(row)

        # Optional section (collapsed by default)
        #opt_common = CollapsibleSection("Optional (output / ckpt)", root, expanded=False)
        #lay_opt.addWidget(opt_common)

        row = QHBoxLayout()
        row.addWidget(QLabel("k:", root))
        self.k_spin = QSpinBox(root); self.k_spin.setRange(1, 64); self.k_spin.setValue(1)
        row.addWidget(self.k_spin)
        row.addWidget(QLabel("half_window:", root))
        self.hw_spin = QSpinBox(root); self.hw_spin.setRange(0, 20); self.hw_spin.setValue(9)
        row.addWidget(self.hw_spin)
        row.addWidget(QLabel("batch_size:", root))
        self.bs_spin = QSpinBox(root); self.bs_spin.setRange(1, 100000); self.bs_spin.setValue(512)
        row.addWidget(self.bs_spin)
        lay_opt.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("output NPY (optional):", root))
        self.output_edit = QLineEdit(root)
        btn_out = QPushButton("Browse", root)
        btn_out.clicked.connect(lambda: self._browse_save_file(self.output_edit, "Save NPY"))
        row.addWidget(self.output_edit, 1)
        row.addWidget(btn_out)
        lay_sel.addLayout(row)

        #row = QHBoxLayout()
        #row.addWidget(QLabel("ckpt (optional):", root))
        #self.ckpt_edit = QLineEdit(root)
        #btn_ckpt = QPushButton("Browse", root)
        #btn_ckpt.clicked.connect(lambda: self._browse_open_file(self.ckpt_edit, "Select ONNX checkpoint"))
        #row.addWidget(self.ckpt_edit, 1)
        #row.addWidget(btn_ckpt)
        #lay_sel.addLayout(row)

        main.addWidget(box_opt)

        # ---- GRID mode ----
        box_grid = QGroupBox("Compute: daqscore compute_grid", root)
        lay_grid = QVBoxLayout(box_grid)

        row = QHBoxLayout()
        row.addWidget(QLabel("contour:", root))
        self.contour_spin = QDoubleSpinBox(root); self.contour_spin.setDecimals(4); self.contour_spin.setRange(-1e9, 1e9); self.contour_spin.setValue(0.0)
        row.addWidget(self.contour_spin)
        row.addWidget(QLabel("stride:", root))
        self.stride_spin = QSpinBox(root); self.stride_spin.setRange(1, 50); self.stride_spin.setValue(2)
        row.addWidget(self.stride_spin)
        row.addWidget(QLabel("max_points:", root))
        self.mp_spin = QSpinBox(root); self.mp_spin.setRange(1000, 100000000); self.mp_spin.setValue(500000)
        row.addWidget(self.mp_spin)
        lay_grid.addLayout(row)

        row = QHBoxLayout()
        self.monitor_chk = QCheckBox("monitor", root)
        self.monitor_chk.setChecked(False)
        row.addWidget(self.monitor_chk)
        row.addStretch(1)
        lay_grid.addLayout(row)

        btn_grid = QPushButton("Run compute_grid", root)
        btn_grid.clicked.connect(self._run_compute_grid)
        lay_grid.addWidget(btn_grid)

        main.addWidget(box_grid)

        # ---- PDB mode ----
        box_pdb = QGroupBox("Compute: daqscore compute_pdb", root)
        lay_pdb = QVBoxLayout(box_pdb)

        row = QHBoxLayout()
        self.apply_color_chk = QCheckBox("apply_color", root); self.apply_color_chk.setChecked(True)
        row.addWidget(self.apply_color_chk)
        row.addStretch(1)
        lay_pdb.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("save_model (optional):", root))
        self.save_model_edit = QLineEdit(root)
        btn_save_model = QPushButton("Browse", root)
        btn_save_model.clicked.connect(lambda: self._browse_save_file(self.save_model_edit, "Save scored model (PDB/CIF)"))
        row.addWidget(self.save_model_edit, 1)
        row.addWidget(btn_save_model)
        lay_pdb.addLayout(row)

        btn_pdb = QPushButton("Run compute_pdb", root)
        btn_pdb.clicked.connect(self._run_compute_pdb)
        lay_pdb.addWidget(btn_pdb)

        main.addWidget(box_pdb)

        # ---- Color-only ----
        box_col = QGroupBox("Color only: daqcolor apply/monitor (use existing NPY)", root)
        lay_col = QVBoxLayout(box_col)

        row = QHBoxLayout()
        row.addWidget(QLabel("npy_path:", root))
        self.npy_edit = QLineEdit(root)
        btn_npy = QPushButton("Browse", root)
        btn_npy.clicked.connect(lambda: self._browse_open_file(self.npy_edit, "Select NPY (N×32)"))
        row.addWidget(self.npy_edit, 1)
        row.addWidget(btn_npy)
        lay_col.addLayout(row)

        # clamp defaults: -1.0 .. 1.0
        row = QHBoxLayout()
        row.addWidget(QLabel("clamp_min:", root))
        self.cmin_edit = QLineEdit("-1.0", root)
        row.addWidget(self.cmin_edit)
        row.addWidget(QLabel("clamp_max:", root))
        self.cmax_edit = QLineEdit("1.0", root)
        row.addWidget(self.cmax_edit)
        lay_col.addLayout(row)

        # Optional: monitor
        #opt_col = CollapsibleSection("Optional (monitor)", root, expanded=False)
        #lay_col.addWidget(opt_col)

        row = QHBoxLayout()
        self.color_monitor_chk = QCheckBox("monitor (daqcolor monitor)", root)
        self.color_monitor_chk.setChecked(False)
        row.addWidget(self.color_monitor_chk)
        #row.addStretch(1)
        #lay_col.addLayout(row)

        #row = QHBoxLayout()
        row.addWidget(QLabel("interval (sec):", root))
        self.color_monitor_interval = QDoubleSpinBox(root)
        self.color_monitor_interval.setDecimals(2)
        self.color_monitor_interval.setRange(0.05, 10.0)
        self.color_monitor_interval.setValue(0.50)
        row.addWidget(self.color_monitor_interval)
        #row.addStretch(1)
        lay_col.addLayout(row)

        # Buttons
        btn_color = QPushButton("Apply coloring", root)
        btn_color.clicked.connect(self._run_color_apply)
        lay_col.addWidget(btn_color)

        btn_monitor = QPushButton("Apply + Monitor (toggle)", root)
        btn_monitor.clicked.connect(self._run_color_monitor_toggle)
        lay_col.addWidget(btn_monitor)

        btn_clear = QPushButton("Clear point models (daqcolor clear)", root)
        btn_clear.clicked.connect(lambda: run(self.session, "daqcolor clear"))
        lay_col.addWidget(btn_clear)

        main.addWidget(box_col)


        # finalize
        self.tool_window.ui_area.setLayout(QVBoxLayout())
        self.tool_window.ui_area.layout().addWidget(root)

        # initial refresh
        self._refresh_models()

    # ---------------- Command runners ----------------
    def _run_compute_grid(self):
        map_tok = self._map_input_token()
        if map_tok is None:
            self.session.logger.error("Select a loaded map or specify a map file path.")
            return

        contour = float(self.contour_spin.value())

        cmd = f"daqscore compute_grid {map_tok} {contour}"

        st_tok = self._structure_token_or_none()
        if st_tok is not None:
            cmd += f" structure {st_tok}"

        # keywords
        cmd += f" stride {int(self.stride_spin.value())}"
        cmd += f" batch_size {int(self.bs_spin.value())}"
        cmd += f" max_points {int(self.mp_spin.value())}"
        cmd += f" k {int(self.k_spin.value())}"
        cmd += f" half_window {int(self.hw_spin.value())}"

        metric = self.metric_edit.text().strip()
        if metric:
            cmd += f" metric \"{metric}\""

        outp = self.output_edit.text().strip()
        if outp:
            cmd += f" output \"{outp}\""

        ckpt = self.ckpt_edit.text().strip()
        if ckpt:
            cmd += f" ckpt \"{ckpt}\""

        cmd += f" monitor {str(bool(self.monitor_chk.isChecked())).lower()}"

        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd)

    def _run_compute_pdb(self):
        map_tok = self._map_input_token()
        if map_tok is None:
            self.session.logger.error("Select a loaded map or specify a map file path.")
            return

        st_tok = self._structure_token_or_none()
        if st_tok is None:
            self.session.logger.error("compute_pdb requires a Structure.")
            return

        cmd = f"daqscore compute_pdb {map_tok} structure {st_tok}"

        cmd += f" batch_size {int(self.bs_spin.value())}"
        cmd += f" k {int(self.k_spin.value())}"
        cmd += f" half_window {int(self.hw_spin.value())}"

        metric = self.metric_edit.text().strip()
        if metric:
            cmd += f" metric \"{metric}\""

        outp = self.output_edit.text().strip()
        if outp:
            cmd += f" output \"{outp}\""

        ckpt = self.ckpt_edit.text().strip()
        if ckpt:
            cmd += f" ckpt \"{ckpt}\""

        cmd += f" apply_color {str(bool(self.apply_color_chk.isChecked())).lower()}"

        save_model = self.save_model_edit.text().strip()
        if save_model:
            cmd += f" save_model \"{save_model}\""

        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd)

    def _run_color_apply(self):
        st_tok = self._structure_token_or_none()
        if st_tok is None:
            self.session.logger.error("Select a Structure to color.")
            return

        npy = self.npy_edit.text().strip()
        if not npy:
            self.session.logger.error("Specify npy_path.")
            return

        cmd = f"daqcolor apply \"{npy}\" {st_tok}"
        cmd += f" k {int(self.k_spin.value())}"
        cmd += f" half_window {int(self.hw_spin.value())}"

        metric = self.metric_edit.text().strip()
        if metric:
            cmd += f" metric \"{metric}\""

        # optional clamp
        cmin = self.cmin_edit.text().strip()
        cmax = self.cmax_edit.text().strip()
        if cmin:
            cmd += f" clamp_min {float(cmin)}"
        if cmax:
            cmd += f" clamp_max {float(cmax)}"

        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd)

    def _run_color_monitor_toggle(self):
        st_tok = self._structure_token_or_none()
        if st_tok is None:
            self.session.logger.error("Select a Structure to color/monitor.")
            return

        npy = self.npy_edit.text().strip()
        if not npy:
            self.session.logger.error("Specify npy_path.")
            return

        # on/off
        on = bool(self.color_monitor_chk.isChecked())
        interval = float(self.color_monitor_interval.value())

        cmd = f"daqcolor monitor {st_tok}"

        # turning on requires npy_path in your cmd.py
        if on:
            cmd += f" npy_path \"{npy}\""
        cmd += f" on {str(on).lower()}"
        cmd += f" interval {interval:.2f}"

        # shared options
        cmd += f" k {int(self.k_spin.value())}"
        cmd += f" half_window {int(self.hw_spin.value())}"

        metric = self.metric_combo.currentText() if hasattr(self, "metric_combo") else ""
        if metric:
            cmd += f" metric \"{metric}\""

        # NOTE: daqcolor_monitor currently does not accept clamp_min/max in your cmd.py
        # so we do not pass clamp_* here.

        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd)
