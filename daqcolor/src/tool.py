# src/tool.py
import os
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

from Qt.QtGui import QDesktopServices, QPixmap
from Qt.QtCore import QUrl, QTimer


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
        
        # Set up auto-refresh handlers for model changes
        self._model_add_handler = session.triggers.add_handler('add models', self._on_models_changed)
        self._model_remove_handler = session.triggers.add_handler('remove models', self._on_models_changed)
        
        self.tool_window.manage(None)

    def delete(self):
        """Clean up handlers when tool is closed."""
        if hasattr(self, '_model_add_handler'):
            self.session.triggers.remove_handler(self._model_add_handler)
        if hasattr(self, '_model_remove_handler'):
            self.session.triggers.remove_handler(self._model_remove_handler)
        super().delete()

    def _on_models_changed(self, trigger_name, data):
        """Called when models are added or removed."""
        self._refresh_models()

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
        # Save current selections
        current_structure = self.structure_combo.currentData()
        current_volume = self.volume_combo.currentData()
        
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

        structure_index = -1
        volume_index = -1
        
        for m in self.session.models.list():
            if Structure is not None and isinstance(m, Structure):
                self.structure_combo.addItem(f"#{m.id_string} {m.name}", m)
                if m == current_structure:
                    structure_index = self.structure_combo.count() - 1
            if Volume is not None and isinstance(m, Volume):
                self.volume_combo.addItem(f"#{m.id_string} {m.name}", m)
                if m == current_volume:
                    volume_index = self.volume_combo.count() - 1
        
        # Restore selections if models still exist
        if structure_index >= 0:
            self.structure_combo.setCurrentIndex(structure_index)
        if volume_index >= 0:
            self.volume_combo.setCurrentIndex(volume_index)

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
            self.session.logger.error(f"{context}: Output/Load Existing NPY path must be specified.")
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
        msg.setText("The specified NPY file already exists.\n Do you want to overwrite it?")
        msg.setInformativeText(path)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        return msg.exec() == QMessageBox.Yes

    # ---------------- Contour auto-sync ----------------
    def _get_displayed_contour_from_selected_volume(self):
        vol = self._selected_volume()
        if vol is None:
            return None
        if hasattr(vol, "surfaces") and vol.surfaces:
            try:
                return float(vol.surfaces[0].level)
            except Exception:
                return None
        return None

    def _on_contour_spin_changed_by_user(self, _v):
        # プログラム側更新中に user override が立たないように guard
        if getattr(self, "_contour_programmatic_update", False):
            return
        self._contour_user_override = True

    def _sync_contour_from_map_display(self):
        # skip if user override
        if getattr(self, "_contour_user_override", False):
            return

        disp = self._get_displayed_contour_from_selected_volume()
        if disp is None:
            return

        # do not change if same
        if abs(self.contour_spin.value() - disp) < 1e-6:
            return

        # signal
        self._contour_programmatic_update = True
        try:
            self.contour_spin.blockSignals(True)
            self.contour_spin.setValue(disp)
        finally:
            self.contour_spin.blockSignals(False)
            self._contour_programmatic_update = False


    # ---------------- Build UI ----------------
    def _build_ui(self):
        parent = self.tool_window.ui_area
        root = QWidget(parent)
        main = QVBoxLayout(root)

        # ==============================
        # Manual link with icon (TOP)
        # ==============================
        manual_url = "https://cxtoolshed.rbvi.ucsf.edu/apps/chimeraxdaqplugin"

        link_row = QHBoxLayout()

        # --- icon ---
        #icon_label = QLabel(root)
        #icon = QPixmap(":/icons/help.png")  # ChimeraX built-in help icon
        #icon_label.setPixmap(icon.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        #link_row.addWidget(icon_label)

        # --- text link ---
        text_label = QLabel(
            f'<a href="{manual_url}"><b>DAQplugin User Manual</b></a>',
            root
        )
        text_label.setOpenExternalLinks(False)
        text_label.linkActivated.connect(
            lambda _=None, u=manual_url: QDesktopServices.openUrl(QUrl(u))
        )
        link_row.addWidget(text_label)

        link_row.addStretch(1)
        main.addLayout(link_row)

        # ==============================
        # Hint about tooltips
        # ==============================
        hint_label = QLabel("<i>Hover over parameter labels and buttons to see detailed information</i>", root)
        hint_label.setStyleSheet("color: #666; padding: 5px;")
        main.addWidget(hint_label)

        
        # ---- Model / Map selection ----
        box_sel = QGroupBox("Inputs (Loaded model/Map/NPY files)", root)
        lay_sel = QVBoxLayout(box_sel)

        row = QHBoxLayout()
        structure_label = QLabel("Structure:", root)
        structure_label.setToolTip("Select the atomic structure model to evaluate")
        row.addWidget(structure_label)
        self.structure_combo = QComboBox(root)
        row.addWidget(self.structure_combo, 1)

        map_label = QLabel("Map:", root)
        map_label.setToolTip("Select the cryo-EM density map (volume) for evaluation")
        row.addWidget(map_label)
        self.volume_combo = QComboBox(root)
        row.addWidget(self.volume_combo, 1)

        lay_sel.addLayout(row)

        #Save NPY
        row = QHBoxLayout()
        npy_label = QLabel("Output/Overwrite NPY:", root)
        npy_label.setToolTip("Path to save computed scores in NPY format")
        row.addWidget(npy_label)
        self.output_edit = QLineEdit(root)
        btn_out = QPushButton("Browse", root)
        btn_out.setToolTip("Browse for NPY file location")
        btn_out.clicked.connect(lambda: self._browse_save_file(self.output_edit, "Save NPY"))
        row.addWidget(self.output_edit, 1)
        row.addWidget(btn_out)
        lay_sel.addLayout(row)

        #Load NPY
        row = QHBoxLayout()
        npy_label = QLabel("Load Existing NPY:", root)
        npy_label.setToolTip("Path to load existing scores from NPY file")
        row.addWidget(npy_label)
        self.load_edit = QLineEdit(root)
        btn_load = QPushButton("Browse", root)
        btn_load.setToolTip("Browse for existing NPY file")
        btn_load.clicked.connect(lambda: self._browse_open_file(self.load_edit, "Load NPY"))
        row.addWidget(self.load_edit, 1)
        row.addWidget(btn_load)
        lay_sel.addLayout(row)

        main.addWidget(box_sel)
        

        # ---- Compute options ----
        box_opt = QGroupBox("Compute options", root)
        lay_opt = QVBoxLayout(box_opt)

        row = QHBoxLayout()
        batch_label = QLabel("batch_size:", root)
        batch_label.setToolTip("Number of samples processed in each batch (affects memory usage and speed)")
        row.addWidget(batch_label)
        self.bs_spin = QSpinBox(root); self.bs_spin.setRange(1, 100000); self.bs_spin.setValue(512)
        row.addWidget(self.bs_spin)
        lay_opt.addLayout(row)
        main.addWidget(box_opt)

        # ---- GRID mode ----
        box_grid = QGroupBox("Compute: Grid-based DAQ score computation", root)
        lay_grid = QVBoxLayout(box_grid)

        row = QHBoxLayout()
        contour_label = QLabel("contour:", root)
        contour_label.setToolTip("Density threshold for grid sampling (auto-syncs with map display)")
        row.addWidget(contour_label)
        self.contour_spin = QDoubleSpinBox(root); 
        self.contour_spin.setDecimals(4); 
        self.contour_spin.setRange(-1e9, 1e9); 
        self.contour_spin.setValue(0.0)


        row.addWidget(self.contour_spin)
        stride_label = QLabel("stride:", root)
        stride_label.setToolTip("Sampling interval for grid points (larger = faster but coarser)")
        row.addWidget(stride_label)
        self.stride_spin = QSpinBox(root); self.stride_spin.setRange(1, 50); self.stride_spin.setValue(2)
        row.addWidget(self.stride_spin)
        max_points_label = QLabel("max_points:", root)
        max_points_label.setToolTip("Maximum number of grid points to evaluate (limits computation time)")
        row.addWidget(max_points_label)
        self.mp_spin = QSpinBox(root); self.mp_spin.setRange(1000, 100000000); self.mp_spin.setValue(500000)
        row.addWidget(self.mp_spin)
        lay_grid.addLayout(row)

        # Auto Update contour level ---
        self._contour_user_override = False  # 手入力したら True にする

        # Stop auto update when user changes contour spin
        self.contour_spin.valueChanged.connect(self._on_contour_spin_changed_by_user)

        # Check Map changes 
        self.volume_combo.currentIndexChanged.connect(self._sync_contour_from_map_display)

        # Check displayed contour level every 0.5 sec
        self._contour_timer = QTimer(root)
        self._contour_timer.setInterval(500)  # 0.5 sec
        self._contour_timer.timeout.connect(self._sync_contour_from_map_display)
        self._contour_timer.start()

        # sync once at start
        self._sync_contour_from_map_display()
        # ----


        btn_grid = QPushButton("Run Grid-based DAQ score computation", root)
        btn_grid.setToolTip("Compute DAQ scores on a 3D grid of points around the structure")
        btn_grid.clicked.connect(self._run_compute_grid)
        lay_grid.addWidget(btn_grid)

        main.addWidget(box_grid)

        

        # ---- Color-only ----
        box_col = QGroupBox("Color only: Coloring/Monitoring with existing NPY scores", root)
        lay_col = QVBoxLayout(box_col)

        row = QHBoxLayout()
        npy_use_label = QLabel("npy_path (from Inputs):", root)
        npy_use_label.setToolTip("Path to NPY file containing pre-computed Grid-based DAQ scores (automatically synced from above)")
        row.addWidget(npy_use_label)
        self.npy_use_edit = QLineEdit(root)
        self.npy_use_edit.setReadOnly(True)
        self.npy_use_edit.setPlaceholderText("Use 'Output/Load Existing NPY' specified above")
        row.addWidget(self.npy_use_edit, 1)
        lay_col.addLayout(row)

        self.load_edit.textChanged.connect(self.npy_use_edit.setText)

        # metric, k, half_window options
        row = QHBoxLayout()
        metric_label = QLabel("metric:", root)
        metric_label.setToolTip("Scoring metric: aa_score (Amino-acid-based) or atom_score (C-alpha Atom-based)")
        row.addWidget(metric_label)
        self.metric_combo = QComboBox(root)
        self.metric_combo.addItems([
            "aa_score",    # DAQ(AA)
            "atom_score",  # DAQ(CA)
        ])
        self.metric_combo.setCurrentText("aa_score")
        row.addWidget(self.metric_combo, 1)
 
        k_label = QLabel("k:", root)
        k_label.setToolTip("Number of nearest neighbors for kNN (k-nearest neighbors) local density evaluation")
        row.addWidget(k_label)
        self.k_spin = QSpinBox(root); self.k_spin.setRange(1, 64); self.k_spin.setValue(1)
        row.addWidget(self.k_spin)
        hw_label = QLabel("half_window:", root)
        hw_label.setToolTip("Half-window size for local scoring context (residues on each side) for window averaging")
        row.addWidget(hw_label)
        self.hw_spin = QSpinBox(root); self.hw_spin.setRange(0, 20); self.hw_spin.setValue(9)
        row.addWidget(self.hw_spin)
        lay_col.addLayout(row)

        # clamp defaults: -1.0 .. 1.0
        row = QHBoxLayout()
        cmin_label = QLabel("clamp_min:", root)
        cmin_label.setToolTip("Minimum value for color scale clamping (scores below this value are mapped to blue)")
        row.addWidget(cmin_label)
        self.cmin_edit = QLineEdit("-1.0", root)
        row.addWidget(self.cmin_edit)
        cmax_label = QLabel("clamp_max:", root)
        cmax_label.setToolTip("Maximum value for color scale clamping (scores above this value are mapped to red)")
        row.addWidget(cmax_label)
        self.cmax_edit = QLineEdit("1.0", root)
        row.addWidget(self.cmax_edit)
        

        # Monitor interval
        
        interval_label = QLabel("interval (sec):", root)
        interval_label.setToolTip("Update interval for automatic monitoring (in seconds)")
        row.addWidget(interval_label)
        self.color_monitor_interval = QDoubleSpinBox(root)
        self.color_monitor_interval.setDecimals(2)
        self.color_monitor_interval.setRange(0.05, 10.0)
        self.color_monitor_interval.setValue(0.50)
        row.addWidget(self.color_monitor_interval)
        lay_col.addLayout(row)

        # Buttons
        row = QHBoxLayout()
        btn_apply = QPushButton("Apply coloring", root)
        btn_apply.setToolTip("Color the structure once using existing Grid-based DAQ scores from NPY file")
        btn_apply.clicked.connect(self._run_color_apply)
        row.addWidget(btn_apply)

        btn_start = QPushButton("Start monitor", root)
        btn_start.setToolTip("Start automatic monitoring and coloring with specified update interval")
        btn_start.clicked.connect(lambda: self._run_color_monitor(on=True))
        row.addWidget(btn_start)

        btn_stop = QPushButton("Stop monitor", root)
        btn_stop.setToolTip("Stop automatic monitoring and coloring")
        btn_stop.clicked.connect(lambda: self._run_color_monitor(on=False))
        row.addWidget(btn_stop)
        lay_col.addLayout(row)

        main.addWidget(box_col)

        # ---- PDB mode ----
        box_pdb = QGroupBox("Compute: Structure-based DAQ score computation (Original DAQ style)", root)
        lay_pdb = QVBoxLayout(box_pdb)

        btn_pdb = QPushButton("Run Structure-based DAQ score computation", root)
        btn_pdb.setToolTip("Compute DAQ scores for structure atoms and apply coloring (original DAQ method)")
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
        self.load_edit.setText(outp)

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

        outp = self.output_edit.text().strip()
        if outp:
            cmd += f" output \"{outp}\""

        cmd += f" apply_color true"

        #no save
        #save_model = self.save_model_edit.text().strip()
        #if save_model:
        #    cmd += f" save_model \"{save_model}\""

        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd)
        self.load_edit.setText(outp)

    def _run_color_apply(self):
        st_tok = self._structure_token_or_none()
        if st_tok is None:
            self.session.logger.error("Select a Structure to color.")
            return

        if not self._require_map_and_npy("daqcolor apply"):
            return
        
        #npy = self.output_edit.text().strip()
        npy = self.load_edit.text().strip()

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
        #npy = self.output_edit.text().strip()
        npy = self.load_edit.text().strip()


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
