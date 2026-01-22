# src/tool.py
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDoubleSpinBox,
    QSpinBox, QPushButton, QCheckBox, QGroupBox, QFileDialog, QComboBox,
    QToolButton, QFrame, QSizePolicy, QMessageBox
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
        Use only a loaded ChimeraX Volume model as map_input.
        Returns '#<id>' string or None if not selected.
        """
        vol = self._selected_volume()
        if vol is None:
            return None
        return f"#{vol.id_string}"

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

    # ---------------- Requirements / warnings ----------------
    def _require_map_and_npy(self, context: str) -> bool:
        """
        Enforce: Map & NPY path must always be specified.
        (Requirement #1)
        """
        map_tok = self._map_input_token()
        if map_tok is None:
            self.session.logger.error(f"{context}: Map must be selected (loaded Volume).")
            return False
        npy = self.output_edit.text().strip()
        if not npy:
            self.session.logger.error(f"{context}: Output/Input NPY path must be specified.")
            return False
        return True
    
    def _warn_overwrite_if_exists(self, path: str, title="Overwrite existing file?") -> bool:
        """
        If path exists, warn and ask user to proceed.
        (Requirement #2 for compute_grid)
        """
        try:
            exists = os.path.exists(path)
        except Exception:
            exists = False
        if not exists:
            return True

        msg = QMessageBox(self.tool_window.ui_area)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText("The specified NPY file already exists.\nDo you want to overwrite it?")
        msg.setInformativeText(path)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        return msg.exec() == QMessageBox.Yes

    # ---------------- Build UI ----------------
    def _build_ui(self):
        parent = self.tool_window.ui_area
        root = QWidget(parent)
        main = QVBoxLayout(root)

        # ---- Model / Map selection ----
        box_sel = QGroupBox("Inputs (Loaded model/Map/NPY files)", root)
        lay_sel = QVBoxLayout(box_sel)

        row = QHBoxLayout()
        row.addWidget(QLabel("Structure:", root))
        self.structure_combo = QComboBox(root)
        row.addWidget(self.structure_combo, 1)

        row.addWidget(QLabel("Map:", root))
        self.volume_combo = QComboBox(root)
        row.addWidget(self.volume_combo, 1)

        btn_refresh = QPushButton("Refresh", root)
        btn_refresh.clicked.connect(self._refresh_models)
        row.addWidget(btn_refresh)
        lay_sel.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output/Input NPY:", root))
        self.output_edit = QLineEdit(root)
        btn_out = QPushButton("Browse", root)
        btn_out.clicked.connect(lambda: self._browse_save_file(self.output_edit, "Save NPY"))
        row.addWidget(self.output_edit, 1)
        row.addWidget(btn_out)
        lay_sel.addLayout(row)

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


        btn_grid = QPushButton("Run compute_grid", root)
        btn_grid.clicked.connect(self._run_compute_grid)
        lay_grid.addWidget(btn_grid)

        main.addWidget(box_grid)

        

        # ---- Color-only ----
        box_col = QGroupBox("Color only: daqcolor apply/monitor (use existing NPY)", root)
        lay_col = QVBoxLayout(box_col)

        row = QHBoxLayout()
        row.addWidget(QLabel("npy_path (from Inputs):", root))
        self.npy_use_edit = QLineEdit(root)
        self.npy_use_edit.setReadOnly(True)
        self.npy_use_edit.setPlaceholderText("Use 'Output/Input NPY' specified above")
        row.addWidget(self.npy_use_edit, 1)
        lay_col.addLayout(row)

        self.output_edit.textChanged.connect(self.npy_use_edit.setText)

        # clamp defaults: -1.0 .. 1.0
        row = QHBoxLayout()
        row.addWidget(QLabel("clamp_min:", root))
        self.cmin_edit = QLineEdit("-1.0", root)
        row.addWidget(self.cmin_edit)
        row.addWidget(QLabel("clamp_max:", root))
        self.cmax_edit = QLineEdit("1.0", root)
        row.addWidget(self.cmax_edit)
        

        # Monitor interval
        
        row.addWidget(QLabel("interval (sec):", root))
        self.color_monitor_interval = QDoubleSpinBox(root)
        self.color_monitor_interval.setDecimals(2)
        self.color_monitor_interval.setRange(0.05, 10.0)
        self.color_monitor_interval.setValue(0.50)
        row.addWidget(self.color_monitor_interval)
        lay_col.addLayout(row)

        # Buttons
        row = QHBoxLayout()
        btn_apply = QPushButton("Apply coloring", root)
        btn_apply.clicked.connect(self._run_color_apply)
        row.addWidget(btn_apply)

        btn_start = QPushButton("Start monitor", root)
        btn_start.clicked.connect(lambda: self._run_color_monitor(on=True))
        row.addWidget(btn_start)

        btn_stop = QPushButton("Stop monitor", root)
        btn_stop.clicked.connect(lambda: self._run_color_monitor(on=False))
        row.addWidget(btn_stop)
        lay_col.addLayout(row)

        main.addWidget(box_col)

        # ---- PDB mode ----
        box_pdb = QGroupBox("Compute: daqscore compute_pdb (original DAQ style)", root)
        lay_pdb = QVBoxLayout(box_pdb)

        row = QHBoxLayout()
        self.apply_color_chk = QCheckBox("apply_color", root); self.apply_color_chk.setChecked(True)
        row.addWidget(self.apply_color_chk)
        row.addStretch(1)
        lay_pdb.addLayout(row)

        btn_pdb = QPushButton("Run compute_pdb", root)
        btn_pdb.clicked.connect(self._run_compute_pdb)
        lay_pdb.addWidget(btn_pdb)

        main.addWidget(box_pdb)

        # finalize
        self.tool_window.ui_area.setLayout(QVBoxLayout())
        self.tool_window.ui_area.layout().addWidget(root)

        # initial refresh
        self._refresh_models()
        self.npy_use_edit.setText(self.output_edit.text())

    # ---------------- Command runners ----------------
    def _run_compute_grid(self):
        if not self._require_map_and_npy("compute_grid"):
            return
        map_tok = self._map_input_token()
        outp = self.output_edit.text().strip()

        if not self._warn_overwrite_if_exists(outp, title="Overwrite NPY from compute_grid?"):
            self.session.logger.info("compute_grid canceled by user (overwrite declined).")
            return
        
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

        metric = self.metric_combo.currentText().strip()
        if metric:
            cmd += f" metric \"{metric}\""

        outp = self.output_edit.text().strip()
        if outp:
            cmd += f" output \"{outp}\""

        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd)

    def _run_compute_pdb(self):
        if not self._require_map_and_npy("compute_pdb"):
            return
        
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

        metric = self.metric_combo.currentText().strip()
        if metric:
            cmd += f" metric \"{metric}\""

        #outp = self.output_edit.text().strip()
        #if outp:
        #    cmd += f" output \"{outp}\""

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

        if not self._require_map_and_npy("daqcolor apply"):
            return
        
        npy = self.output_edit.text().strip()
        
        if not npy:
            self.session.logger.error("Specify npy_path.")
            return

        cmd = f"daqcolor apply \"{npy}\" {st_tok}"
        cmd += f" k {int(self.k_spin.value())}"
        cmd += f" half_window {int(self.hw_spin.value())}"

        metric = self.metric_combo.currentText().strip()
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

    def _run_color_monitor(self, on: bool):
        # Requirement #3: Structure must be selected in Inputs
        st_tok = self._structure_token_or_none()
        if st_tok is None:
            self.session.logger.error("Color-only monitor requires a Structure (select in Inputs).")
            return

        # Requirement #1: Map & NPY are mandatory (use NPY from Inputs)
        if not self._require_map_and_npy("daqcolor monitor"):
            return
        npy = self.output_edit.text().strip()

        interval = float(self.color_monitor_interval.value())

        cmd = f"daqcolor monitor {st_tok}"

        # turning on requires npy_path
        if on:
            cmd += f" npy_path \"{npy}\""
        cmd += f" on {str(on).lower()}"
        cmd += f" interval {interval:.2f}"

        # shared options
        cmd += f" k {int(self.k_spin.value())}"
        cmd += f" half_window {int(self.hw_spin.value())}"

        metric = self.metric_combo.currentText().strip()
        if metric:
            cmd += f" metric \"{metric}\""

        # optional clamp (apply only)
        cmin = self.cmin_edit.text().strip()
        cmax = self.cmax_edit.text().strip()
        if cmin:
            cmd += f" clamp_min {float(cmin)}"
        if cmax:
            cmd += f" clamp_max {float(cmax)}"

        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd)
