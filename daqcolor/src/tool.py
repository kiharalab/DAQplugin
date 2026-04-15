# src/tool.py
import os
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDoubleSpinBox,
    QSpinBox, QPushButton, QCheckBox, QGroupBox, QFileDialog, QComboBox,
    QToolButton, QFrame, QSizePolicy, QMessageBox, QGridLayout, QTabWidget
)

from Qt.QtCore import Qt

from Qt.QtGui import QDesktopServices
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

        self._arrowwin_group = None  # To track the ArrowWin group model for easy removal

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

    def _selected_metric(self):
        metric = self.metric_combo.currentData()
        if metric is None:
            return None
        return str(metric).strip()

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
            self.session.logger.error(f"{context}: Output NPY path must be specified.")
            return False
        return True
    
    def _require_npy(self, context: str) -> bool:
        """
        Enforce: Load NPY path must always be specified.
        """
        npy = self.load_edit.text().strip()
        if not npy:
            self.session.logger.error(f"{context}: Load NPY path must be specified.")
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
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        root.setStyleSheet("""
            QWidget {
                background: #000000;
                color: #ffffff;
                font-family: "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 13px;
            }
            QWidget[section="dark"] {
                background: #000000;
                color: #ffffff;
            }
            QWidget[section="light"] {
                background: #1a1a1c;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: none;
                background: transparent;
                top: 0px;
            }
            QTabBar::tab {
                background: rgba(0, 0, 0, 0.8);
                color: rgba(255, 255, 255, 0.92);
                border: none;
                padding: 8px 18px;
                margin-right: 6px;
                min-width: 92px;
                border-radius: 999px;
                font-family: "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 12px;
                font-weight: 400;
            }
            QTabBar::tab:selected {
                background: #1d1d1f;
                color: #ffffff;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background: rgba(0, 0, 0, 0.88);
            }
            QGroupBox {
                border: none;
                margin-top: 6px;
                padding: 12px 12px 12px 12px;
                border-radius: 10px;
                font-family: "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 15px;
                font-weight: 600;
                line-height: 1.24;
            }
            QGroupBox[card="light"] {
                background: #ffffff;
                color: #1d1d1f;
            }
            QGroupBox[card="dark"] {
                background: #1d1d1f;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 2px;
            }
            QFrame[card="light"], QFrame[card="dark"] {
                border: none;
                border-radius: 10px;
            }
            QFrame[card="light"] {
                background: #272729;
            }
            QFrame[card="dark"] {
                background: #1d1d1f;
            }
            QLabel {
                background: transparent;
                color: inherit;
                font-family: "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 14px;
                font-weight: 400;
            }
            QLabel[role="field-label"] {
                font-family: "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 13px;
                font-weight: 600;
            }
            QFrame[card="dark"] QLabel,
            QGroupBox[card="dark"] QLabel,
            QWidget[section="dark"] QLabel {
                color: #ffffff;
            }
            QFrame[card="light"] QLabel,
            QGroupBox[card="light"] QLabel,
            QWidget[section="light"] QLabel {
                color: #ffffff;
            }
            QLabel[role="title"] {
                font-family: "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 20px;
                font-weight: 600;
                line-height: 1.14;
                letter-spacing: 0.2px;
            }
            QLabel[role="caption"] {
                font-size: 12px;
                color: rgba(255, 255, 255, 0.72);
            }
            QWidget#sectionContent {
                background: transparent;
            }
            QWidget[section="dark"] QLabel[role="caption"] {
                color: rgba(255, 255, 255, 0.72);
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                color: #1d1d1f;
                border: none;
                border-radius: 12px;
                padding: 6px 10px;
                min-height: 22px;
                selection-background-color: #0071e3;
                selection-color: #ffffff;
                font-family: "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 13px;
            }
            QWidget[section="dark"] QLineEdit,
            QWidget[section="dark"] QComboBox,
            QWidget[section="dark"] QSpinBox,
            QWidget[section="dark"] QDoubleSpinBox {
                background: #272729;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 24px;
            }
            QPushButton {
                background: #ffffff;
                color: #1d1d1f;
                border: 1px solid rgba(0, 0, 0, 0.08);
                border-radius: 14px;
                padding: 6px 13px;
                min-height: 22px;
                font-family: "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 13px;
                font-weight: 400;
            }
            QPushButton:hover {
                background: #fafafc;
            }
            QPushButton:focus {
                outline: none;
                border: 2px solid #0071e3;
            }
            QPushButton[variant="primary"] {
                background: #0071e3;
                color: #ffffff;
                border: 1px solid transparent;
            }
            QPushButton[variant="primary"]:hover {
                background: #0077ed;
            }
            QPushButton[variant="secondary-dark"] {
                background: #2a2a2d;
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.22);
            }
            QPushButton[variant="secondary-dark"]:hover {
                background: #3a3a3d;
            }
            QPushButton[variant="secondary-gray"] {
                background: #3a3a3c;
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.14);
            }
            QPushButton[variant="secondary-gray"]:hover {
                background: #4a4a4d;
            }
            QPushButton[variant="secondary-gray"]:checked {
                background: #5a5a5f;
                border: 1px solid rgba(255, 255, 255, 0.28);
            }
            QPushButton[variant="pill-link"] {
                background: transparent;
                color: #0066cc;
                border: 1px solid #0066cc;
            }
            QWidget[section="dark"] QPushButton[variant="pill-link"] {
                color: #2997ff;
                border: 1px solid #2997ff;
            }
            QPushButton[variant="pill-link"]:hover {
                text-decoration: underline;
            }
            QPushButton[variant="outline-light"] {
                background: #3a3a3c;
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.14);
            }
            QPushButton[variant="outline-light"]:hover {
                background: #4a4a4d;
            }
            QCheckBox {
                spacing: 8px;
                background: transparent;
                font-size: 14px;
            }
            QWidget[section="dark"] QCheckBox {
                color: #ffffff;
            }
        """)

        def make_field_box(title: str, card: str = "light", layout_cls=QVBoxLayout):
            box = QFrame(root)
            box.setProperty("card", card)
            outer_layout = QVBoxLayout(box)
            outer_layout.setContentsMargins(12, 10, 12, 10)
            outer_layout.setSpacing(6)

            label = QLabel(title, box)
            label.setProperty("role", "field-label")
            outer_layout.addWidget(label)

            content = QWidget(box)
            content.setObjectName("sectionContent")
            layout = layout_cls(content)
            if isinstance(layout, QGridLayout):
                layout.setContentsMargins(0, 0, 0, 0)
            else:
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(6)
            outer_layout.addWidget(content)

            return box, layout, label

        def make_button(text: str, tooltip: str, slot, variant: str = "primary", min_width: int = 180):
            button = QPushButton(text, root)
            button.setProperty("variant", variant)
            button.setToolTip(tooltip)
            button.clicked.connect(slot)
            button.setMinimumWidth(min_width)
            button.style().unpolish(button)
            button.style().polish(button)
            return button

        tabs = QTabWidget(root)
        outer.addWidget(tabs, 1)

        main_tab = QWidget(root)
        main_tab.setProperty("section", "dark")
        main_layout = QVBoxLayout(main_tab)
        main_layout.setContentsMargins(12, 10, 12, 12)
        main_layout.setSpacing(8)

        params_tab = QWidget(root)
        params_tab.setProperty("section", "dark")
        params_layout = QVBoxLayout(params_tab)
        params_layout.setContentsMargins(12, 10, 12, 12)
        params_layout.setSpacing(8)

        tabs.addTab(main_tab, "MAIN")
        tabs.addTab(params_tab, "Parameters")

        # ---- Main tab: inputs ----
        input_grid = QGridLayout()
        input_grid.setHorizontalSpacing(10)
        input_grid.setVerticalSpacing(8)

        main_header = QLabel("DAQplugin", root)
        main_header.setProperty("role", "title")
        main_layout.addWidget(main_header)

        structure_box, structure_layout, structure_label = make_field_box("Structure", card="dark")
        structure_box.setToolTip("Select the atomic structure model to evaluate")
        structure_label.setToolTip("Select the atomic structure model to evaluate")
        self.structure_combo = QComboBox(root)
        self.structure_combo.setToolTip("Select the atomic structure model to evaluate")
        structure_layout.addWidget(self.structure_combo)
        input_grid.addWidget(structure_box, 0, 0)

        map_box, map_layout, map_label = make_field_box("Map", card="dark")
        map_box.setToolTip("Select the cryo-EM density map (volume) for evaluation")
        map_label.setToolTip("Select the cryo-EM density map (volume) for evaluation")
        self.volume_combo = QComboBox(root)
        self.volume_combo.setToolTip("Select the cryo-EM density map (volume) for evaluation")
        map_layout.addWidget(self.volume_combo)
        input_grid.addWidget(map_box, 0, 1)

        output_box, output_layout, output_label = make_field_box("Output NPY", card="dark")
        output_box.setToolTip("Path to save computed scores in NPY format")
        output_label.setToolTip("Path to save computed scores in NPY format")
        self.output_edit = QLineEdit(root)
        self.output_edit.setToolTip("Path to save computed scores in NPY format")
        btn_out = QPushButton("Choose File", root)
        btn_out.setProperty("variant", "pill-link")
        btn_out.setToolTip("Browse for NPY file location")
        btn_out.clicked.connect(lambda: self._browse_save_file(self.output_edit, "Save NPY"))
        btn_out.style().unpolish(btn_out)
        btn_out.style().polish(btn_out)
        output_row = QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.setSpacing(8)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(btn_out)
        output_layout.addLayout(output_row)
        input_grid.addWidget(output_box, 1, 0)

        load_box, load_layout, load_label = make_field_box("Load NPY", card="dark")
        load_box.setToolTip("Path to load existing scores from NPY file")
        load_label.setToolTip("Path to load existing scores from NPY file")
        self.load_edit = QLineEdit(root)
        self.load_edit.setToolTip("Path to load existing scores from NPY file")
        btn_load = QPushButton("Choose File", root)
        btn_load.setProperty("variant", "pill-link")
        btn_load.setToolTip("Browse for existing NPY file")
        btn_load.clicked.connect(lambda: self._browse_open_file(self.load_edit, "Load NPY"))
        btn_load.style().unpolish(btn_load)
        btn_load.style().polish(btn_load)
        load_row = QHBoxLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.setSpacing(8)
        load_row.addWidget(self.load_edit, 1)
        load_row.addWidget(btn_load)
        load_layout.addLayout(load_row)
        input_grid.addWidget(load_box, 1, 1)

        main_layout.addLayout(input_grid)

        run_daq_group, run_daq_layout, _ = make_field_box("Grid-based DAQ", card="dark", layout_cls=QHBoxLayout)
        run_daq_layout.setSpacing(10)

        btn_grid = make_button(
            "Calculate DAQ Scores",
            "Compute DAQ scores on a 3D grid of points around the structure",
            self._run_compute_grid,
            variant="primary",
            min_width=160,
        )
        run_daq_layout.addWidget(btn_grid, 0, Qt.AlignLeft)
        run_daq_layout.addStretch(1)
        main_layout.addWidget(run_daq_group)

        metric_box, metric_layout, metric_label = make_field_box("Metric", card="dark")
        metric_box.setToolTip("Scoring metric: aa_score (Amino-acid-based) or atom_score (C-alpha Atom-based)")
        metric_label.setToolTip("Scoring metric: aa_score (Amino-acid-based) or atom_score (C-alpha Atom-based)")
        self.metric_combo = QComboBox(root)
        self.metric_combo.setToolTip("Scoring metric: aa_score (Amino-acid-based) or atom_score (C-alpha Atom-based)")
        self.metric_combo.addItem("DAQ(AA)", "aa_score")
        self.metric_combo.addItem("DAQ(Ca)", "atom_score")
        self.metric_combo.addItem("DAQ(SS)", "ss_score")
        self.metric_combo.setCurrentIndex(0)
        metric_layout.addWidget(self.metric_combo)
        main_layout.addWidget(metric_box)

        color_group, color_layout, _ = make_field_box("Coloring / Monitoring", card="dark", layout_cls=QHBoxLayout)
        color_layout.setSpacing(10)

        btn_apply = make_button(
            "Color Structure",
            "Color the structure once using existing Grid-based DAQ scores from NPY file",
            self._run_color_apply,
            variant="primary",
            min_width=140,
        )
        color_layout.addWidget(btn_apply)

        btn_start = make_button(
            "Start Live Update",
            "Start automatic monitoring and coloring with specified update interval",
            lambda: self._run_color_monitor(on=True),
            variant="primary",
            min_width=155,
        )
        color_layout.addWidget(btn_start)

        btn_stop = make_button(
            "Stop Update",
            "Stop automatic monitoring and coloring",
            lambda: self._run_color_monitor(on=False),
            variant="secondary-gray",
            min_width=88,
        )
        color_layout.addWidget(btn_stop)
        main_layout.addWidget(color_group)

        arrow_group, arrow_layout, _ = make_field_box("Sequence Shift Suggestions", card="dark")
        arrow_layout.setSpacing(8)

        arrow_button_grid = QGridLayout()
        arrow_button_grid.setHorizontalSpacing(10)
        arrow_button_grid.setVerticalSpacing(10)
        arrow_button_grid.setContentsMargins(0, 0, 0, 0)

        btn_arrow = make_button(
            "Show Shift Arrows",
            "Draw backbone-shift suggestion cones. If residues are selected, only selected residues are processed; otherwise the whole model is processed",
            lambda: self._run_arrowwin(apply_constraints=False),
            variant="primary",
            min_width=150,
        )
        arrow_button_grid.addWidget(btn_arrow, 0, 0)

        btn_arrow_clear = make_button(
            "Clear Shift Arrows",
            "Remove all ArrowWin cones by deleting the group model",
            self._clear_arrowwin_group,
            variant="secondary-gray",
            min_width=150,
        )
        arrow_button_grid.addWidget(btn_arrow_clear, 0, 1)

        btn_add_constraints = make_button(
            "Add Arrow Constraints",
            "Draw arrows and also add ISOLDE position restraints based on residue mapping",
            lambda: self._run_arrowwin(apply_constraints=True),
            variant="primary",
            min_width=160,
        )
        arrow_button_grid.addWidget(btn_add_constraints, 1, 0)

        btn_clear_rest = make_button(
            "Clear Arrow Constraints",
            "Disable DAQ-created ISOLDE position restraints (only those created by DAQ arrowwin)",
            self._clear_daq_restraints,
            variant="secondary-gray",
            min_width=170,
        )
        arrow_button_grid.addWidget(btn_clear_rest, 1, 1)
        arrow_layout.addLayout(arrow_button_grid)
        main_layout.addWidget(arrow_group)

        atom_group, atom_layout, _ = make_field_box("Atom Position Based DAQ", card="dark", layout_cls=QHBoxLayout)
        atom_layout.setSpacing(10)

        btn_pdb = make_button(
            "Calculate Atom-Based DAQ",
            "Compute DAQ scores for structure atoms and apply coloring (original DAQ method)",
            self._run_compute_pdb,
            variant="primary",
            min_width=250,
        )
        atom_layout.addWidget(btn_pdb, 0, Qt.AlignLeft)
        atom_layout.addStretch(1)
        main_layout.addWidget(atom_group)
        main_layout.addStretch(1)

        # ---- Parameters tab ----
        params_header = QLabel("Parameters", root)
        params_header.setProperty("role", "title")
        params_layout.addWidget(params_header)

        compute_group, compute_layout, _ = make_field_box("Compute Settings", card="dark", layout_cls=QGridLayout)
        compute_layout.setHorizontalSpacing(10)
        compute_layout.setVerticalSpacing(6)

        batch_label = QLabel("Batch size", root)
        batch_label.setToolTip("Number of samples processed in each batch (affects memory usage and speed)")
        compute_layout.addWidget(batch_label, 0, 0)
        self.bs_spin = QSpinBox(root)
        self.bs_spin.setRange(1, 100000)
        self.bs_spin.setValue(512)
        self.bs_spin.setToolTip("Number of samples processed in each batch (affects memory usage and speed)")
        compute_layout.addWidget(self.bs_spin, 0, 1)
        params_layout.addWidget(compute_group)

        grid_group, grid_layout, _ = make_field_box("Grid Settings", card="dark", layout_cls=QGridLayout)
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(6)

        contour_label = QLabel("Contour", root)
        contour_label.setToolTip("Density threshold for grid sampling (auto-syncs with map display)")
        grid_layout.addWidget(contour_label, 0, 0)
        self.contour_spin = QDoubleSpinBox(root)
        self.contour_spin.setDecimals(4)
        self.contour_spin.setRange(-1e9, 1e9)
        self.contour_spin.setValue(0.0)
        self.contour_spin.setToolTip("Density threshold for grid sampling (auto-syncs with map display)")
        grid_layout.addWidget(self.contour_spin, 0, 1)

        stride_label = QLabel("Stride", root)
        stride_label.setToolTip("Sampling interval for grid points (larger = faster but coarser)")
        grid_layout.addWidget(stride_label, 0, 2)
        self.stride_spin = QSpinBox(root)
        self.stride_spin.setRange(1, 50)
        self.stride_spin.setValue(2)
        self.stride_spin.setToolTip("Sampling interval for grid points (larger = faster but coarser)")
        grid_layout.addWidget(self.stride_spin, 0, 3)

        max_points_label = QLabel("Max Points", root)
        max_points_label.setToolTip("Maximum number of grid points to evaluate (limits computation time)")
        grid_layout.addWidget(max_points_label, 1, 0)
        self.mp_spin = QSpinBox(root)
        self.mp_spin.setRange(1000, 100000000)
        self.mp_spin.setValue(500000)
        self.mp_spin.setToolTip("Maximum number of grid points to evaluate (limits computation time)")
        grid_layout.addWidget(self.mp_spin, 1, 1)
        params_layout.addWidget(grid_group)

        # Auto Update contour level ---
        self._contour_user_override = False
        self.contour_spin.valueChanged.connect(self._on_contour_spin_changed_by_user)
        self.volume_combo.currentIndexChanged.connect(self._sync_contour_from_map_display)
        self._contour_timer = QTimer(root)
        self._contour_timer.setInterval(500)
        self._contour_timer.timeout.connect(self._sync_contour_from_map_display)
        self._contour_timer.start()
        self._sync_contour_from_map_display()

        scoring_group, scoring_layout, _ = make_field_box("Scoring Settings", card="dark", layout_cls=QGridLayout)
        scoring_layout.setHorizontalSpacing(10)
        scoring_layout.setVerticalSpacing(6)

        k_label = QLabel("k", root)
        k_label.setToolTip("Number of nearest neighbors for kNN (k-nearest neighbors) local density evaluation")
        scoring_layout.addWidget(k_label, 0, 0)
        self.k_spin = QSpinBox(root)
        self.k_spin.setRange(1, 64)
        self.k_spin.setValue(1)
        self.k_spin.setToolTip("Number of nearest neighbors for kNN (k-nearest neighbors) local density evaluation")
        scoring_layout.addWidget(self.k_spin, 0, 1)

        hw_label = QLabel("Half window", root)
        hw_label.setToolTip("Half-window size for local scoring context (residues on each side) for window averaging")
        scoring_layout.addWidget(hw_label, 0, 2)
        self.hw_spin = QSpinBox(root)
        self.hw_spin.setRange(0, 20)
        self.hw_spin.setValue(9)
        self.hw_spin.setToolTip("Half-window size for local scoring context (residues on each side) for window averaging")
        scoring_layout.addWidget(self.hw_spin, 0, 3)
        params_layout.addWidget(scoring_group)

        color_params_group, color_params_layout, _ = make_field_box("Coloring Settings", card="dark", layout_cls=QGridLayout)
        color_params_layout.setHorizontalSpacing(10)
        color_params_layout.setVerticalSpacing(6)

        cmin_label = QLabel("Clamp min", root)
        cmin_label.setToolTip("Minimum value for color scale clamping (scores below this value are mapped to blue)")
        color_params_layout.addWidget(cmin_label, 0, 0)
        self.cmin_edit = QLineEdit("-1.0", root)
        self.cmin_edit.setToolTip("Minimum value for color scale clamping (scores below this value are mapped to blue)")
        color_params_layout.addWidget(self.cmin_edit, 0, 1)

        cmax_label = QLabel("Clamp max", root)
        cmax_label.setToolTip("Maximum value for color scale clamping (scores above this value are mapped to red)")
        color_params_layout.addWidget(cmax_label, 0, 2)
        self.cmax_edit = QLineEdit("1.0", root)
        self.cmax_edit.setToolTip("Maximum value for color scale clamping (scores above this value are mapped to red)")
        color_params_layout.addWidget(self.cmax_edit, 0, 3)

        interval_label = QLabel("Interval (sec)", root)
        interval_label.setToolTip("Update interval for automatic monitoring (in seconds)")
        color_params_layout.addWidget(interval_label, 1, 0)
        self.color_monitor_interval = QDoubleSpinBox(root)
        self.color_monitor_interval.setDecimals(2)
        self.color_monitor_interval.setRange(0.05, 10.0)
        self.color_monitor_interval.setValue(0.50)
        self.color_monitor_interval.setToolTip("Update interval for automatic monitoring (in seconds)")
        color_params_layout.addWidget(self.color_monitor_interval, 1, 1)
        params_layout.addWidget(color_params_group)

        arrow_params_group, arrow_params_layout, _ = make_field_box("Sequence Shift Suggestion Parameters", card="dark", layout_cls=QGridLayout)
        arrow_params_layout.setHorizontalSpacing(10)
        arrow_params_layout.setVerticalSpacing(6)

        nwin_label = QLabel("Half window", root)
        nwin_label.setToolTip("Half-window size for scoring window around each residue")
        arrow_params_layout.addWidget(nwin_label, 0, 0)
        self.aw_nwin_spin = QSpinBox(root)
        self.aw_nwin_spin.setRange(0, 20)
        self.aw_nwin_spin.setValue(5)
        self.aw_nwin_spin.setToolTip("Half-window size for scoring window around each residue")
        arrow_params_layout.addWidget(self.aw_nwin_spin, 0, 1)

        kshift_label = QLabel("Max shift", root)
        kshift_label.setToolTip("Candidate backbone index shifts in [-kshift..-1, +1..+kshift]")
        arrow_params_layout.addWidget(kshift_label, 0, 2)
        self.aw_kshift_spin = QSpinBox(root)
        self.aw_kshift_spin.setRange(1, 20)
        self.aw_kshift_spin.setValue(5)
        self.aw_kshift_spin.setToolTip("Candidate backbone index shifts in [-kshift..-1, +1..+kshift]")
        arrow_params_layout.addWidget(self.aw_kshift_spin, 0, 3)

        minmove_label = QLabel("Minimum distance", root)
        minmove_label.setToolTip("Minimum Length to draw an arrow (Angstrom)")
        arrow_params_layout.addWidget(minmove_label, 1, 0)
        self.aw_minmove_spin = QDoubleSpinBox(root)
        self.aw_minmove_spin.setDecimals(1)
        self.aw_minmove_spin.setRange(0.0, 10.0)
        self.aw_minmove_spin.setValue(1.0)
        self.aw_minmove_spin.setToolTip("Minimum Length to draw an arrow (Angstrom)")
        arrow_params_layout.addWidget(self.aw_minmove_spin, 1, 1)

        minimp_label = QLabel("Minimum improvement", root)
        minimp_label.setToolTip("Lower bound of average DAQ score improvement (window-mean). Below this, no arrow is drawn")
        arrow_params_layout.addWidget(minimp_label, 1, 2)
        self.aw_minimp_spin = QDoubleSpinBox(root)
        self.aw_minimp_spin.setDecimals(1)
        self.aw_minimp_spin.setRange(0.0, 5.0)
        self.aw_minimp_spin.setValue(0.5)
        self.aw_minimp_spin.setSingleStep(0.1)
        self.aw_minimp_spin.setToolTip("Lower bound of average DAQ score improvement (window-mean). Below this, no arrow is drawn")
        arrow_params_layout.addWidget(self.aw_minimp_spin, 1, 3)

        base_radius_label = QLabel("Base radius", root)
        base_radius_label.setToolTip("Base cone radius (will be scaled by improvement if scaling is enabled)")
        arrow_params_layout.addWidget(base_radius_label, 2, 0)
        self.aw_radius_spin = QDoubleSpinBox(root)
        self.aw_radius_spin.setDecimals(1)
        self.aw_radius_spin.setRange(0.0, 10.0)
        self.aw_radius_spin.setValue(0.4)
        self.aw_radius_spin.setSingleStep(0.1)
        self.aw_radius_spin.setToolTip("Base cone radius (will be scaled by improvement if scaling is enabled)")
        arrow_params_layout.addWidget(self.aw_radius_spin, 2, 1)

        vmax_color_label = QLabel("Max color", root)
        vmax_color_label.setToolTip("Improvement value mapped to maximum redness (>= vmax_color becomes fully red)")
        arrow_params_layout.addWidget(vmax_color_label, 2, 2)
        self.aw_vmax_color_spin = QDoubleSpinBox(root)
        self.aw_vmax_color_spin.setDecimals(1)
        self.aw_vmax_color_spin.setRange(1e-6, 5.0)
        self.aw_vmax_color_spin.setValue(2.0)
        self.aw_vmax_color_spin.setSingleStep(0.1)
        self.aw_vmax_color_spin.setToolTip("Improvement value mapped to maximum redness (>= vmax_color becomes fully red)")
        arrow_params_layout.addWidget(self.aw_vmax_color_spin, 2, 3)

        vmax_radius_label = QLabel("Max radius score", root)
        vmax_radius_label.setToolTip("Improvement value mapped to maximum radius scaling (>= vmax_radius becomes max_radius_scale)")
        arrow_params_layout.addWidget(vmax_radius_label, 3, 0)
        self.aw_vmax_radius_spin = QDoubleSpinBox(root)
        self.aw_vmax_radius_spin.setDecimals(1)
        self.aw_vmax_radius_spin.setRange(1e-6, 5.0)
        self.aw_vmax_radius_spin.setValue(2.0)
        self.aw_vmax_radius_spin.setSingleStep(0.1)
        self.aw_vmax_radius_spin.setToolTip("Improvement value mapped to maximum radius scaling (>= vmax_radius becomes max_radius_scale)")
        arrow_params_layout.addWidget(self.aw_vmax_radius_spin, 3, 1)

        minrs_label = QLabel("Radius scale min", root)
        minrs_label.setToolTip("Radius multiplier at improvement=0")
        arrow_params_layout.addWidget(minrs_label, 3, 2)
        self.aw_minrs_spin = QDoubleSpinBox(root)
        self.aw_minrs_spin.setDecimals(1)
        self.aw_minrs_spin.setRange(0.0, 100.0)
        self.aw_minrs_spin.setValue(0.5)
        self.aw_minrs_spin.setSingleStep(0.1)
        self.aw_minrs_spin.setToolTip("Radius multiplier at improvement=0")
        arrow_params_layout.addWidget(self.aw_minrs_spin, 3, 3)

        maxrs_label = QLabel("Radius scale max", root)
        maxrs_label.setToolTip("Radius multiplier at improvement>=vmax_radius")
        arrow_params_layout.addWidget(maxrs_label, 4, 0)
        self.aw_maxrs_spin = QDoubleSpinBox(root)
        self.aw_maxrs_spin.setDecimals(1)
        self.aw_maxrs_spin.setRange(0.0, 100.0)
        self.aw_maxrs_spin.setValue(2.0)
        self.aw_maxrs_spin.setSingleStep(0.1)
        self.aw_maxrs_spin.setToolTip("Radius multiplier at improvement>=vmax_radius")
        arrow_params_layout.addWidget(self.aw_maxrs_spin, 4, 1)

        spring_label = QLabel("Constraint spring", root)
        spring_label.setToolTip("ISOLDE position restraint spring constant")
        arrow_params_layout.addWidget(spring_label, 4, 2)
        self.aw_spring_spin = QDoubleSpinBox(root)
        self.aw_spring_spin.setDecimals(0)
        self.aw_spring_spin.setRange(0.0, 1e6)
        self.aw_spring_spin.setValue(1500.0)
        self.aw_spring_spin.setSingleStep(100.0)
        self.aw_spring_spin.setToolTip("ISOLDE position restraint spring constant")
        arrow_params_layout.addWidget(self.aw_spring_spin, 4, 3)
        params_layout.addWidget(arrow_params_group)
        params_layout.addStretch(1)

        # ---- Footer help link ----
        manual_url = "https://cxtoolshed.rbvi.ucsf.edu/apps/chimeraxdaqplugin"
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(2, 0, 2, 0)

        footer_hint = QLabel("Hover over controls for details.", root)
        footer_hint.setProperty("role", "caption")
        footer_row.addWidget(footer_hint)
        footer_row.addStretch(1)

        text_label = QLabel(f'<a href="{manual_url}">DAQplugin User Manual</a>', root)
        text_label.setStyleSheet('color: #0066cc; font-size: 14px;')
        text_label.setOpenExternalLinks(False)
        text_label.linkActivated.connect(
            lambda _=None, u=manual_url: QDesktopServices.openUrl(QUrl(u))
        )
        footer_row.addWidget(text_label)
        outer.addLayout(footer_row)

        # finalize
        self.tool_window.ui_area.setLayout(QVBoxLayout())
        self.tool_window.ui_area.layout().addWidget(root)

        # initial refresh
        self._refresh_models()

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

        metric = self._selected_metric()
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

        metric = self._selected_metric()
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

        if not self._require_npy("daqcolor apply"):
            return
        
        #npy = self.output_edit.text().strip()
        npy = self.load_edit.text().strip()

        if not npy:
            self.session.logger.error("Specify npy_path.")
            return

        cmd = f"daqcolor apply \"{npy}\" {st_tok}"
        cmd += f" k {int(self.k_spin.value())}"
        cmd += f" half_window {int(self.hw_spin.value())}"

        metric = self._selected_metric()
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
        if not self._require_npy("daqcolor monitor"):
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

        metric = self._selected_metric()
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

    # ---- ArrowWin ----
    def _run_arrowwin(self, apply_constraints: bool = False):
        """Run backbone-shift suggestion cones (daq arrowwin)."""

        if not self._require_npy("daq arrowwin"):
            return

        npy = self.load_edit.text().strip()
        if not npy:
            self.session.logger.error("ArrowWin requires a NPY file (Load Existing NPY).")
            return

        st_tok = self._structure_token_or_none()
        if st_tok is None:
            self.session.logger.error("ArrowWin requires a Structure selected in Inputs.")
            return

        # NOTE: command itself prioritizes selection if residues are selected.
        cmd = f"daq arrowwin {st_tok} \"{npy}\""

        cmd += f" nwin {int(self.aw_nwin_spin.value())}"
        cmd += f" kshift {int(self.aw_kshift_spin.value())}"
        cmd += f" minmove {float(self.aw_minmove_spin.value()):.3f}"
        cmd += f" radius {float(self.aw_radius_spin.value()):.3f}"
        cmd += f" vmax_color {float(self.aw_vmax_color_spin.value()):.3f}"
        cmd += f" min_improvement {float(self.aw_minimp_spin.value()):.3f}"
        cmd += f" vmax_radius {float(self.aw_vmax_radius_spin.value()):.3f}"
        cmd += f" min_radius_scale {float(self.aw_minrs_spin.value()):.3f}"
        cmd += f" max_radius_scale {float(self.aw_maxrs_spin.value()):.3f}"
        cmd += f" group_name \"DAQ Arrows\""

        # ISOLDE restraints
        if apply_constraints:
            cmd += " apply_isolde_restraints true"
            cmd += f" spring_constant {float(self.aw_spring_spin.value()):.1f}"
        else:
            cmd += " apply_isolde_restraints false"


        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd, log=False)
        # cache group model by name (top-level) after command runs
        self._arrowwin_group = None
        for m in self.session.models.list():
            if (getattr(m, "name", "") or "").strip() == "DAQ Arrows":
                self._arrowwin_group = m
                break

    def _clear_arrowwin_group(self, name: str = "DAQ Arrows"):
        g = getattr(self, "_arrowwin_group", None)
        if g is not None:
            try:
                self.session.models.remove([g])
                self.session.logger.info("Removed ArrowWin group (by reference).")
            except Exception as e:
                self.session.logger.info(f"Failed to remove ArrowWin group: {e}")
            finally:
                self._arrowwin_group = None
            return

        # fallback: remove by name (top-level)
        removed = 0
        for m in list(self.session.models.list()):
            if (getattr(m, "name", "") or "").strip() == name:
                try:
                    self.session.models.remove([m])
                    removed += 1
                except Exception:
                    pass

        if removed:
            self.session.logger.info(f"Removed Arrow group: '{name}' ({removed} model(s))")
        else:
            self.session.logger.info(f"Arrow group not found: '{name}'")

    def _clear_daq_restraints(self):
        """Clear (disable) DAQ-created ISOLDE restraints for selected structure."""
        st_tok = self._structure_token_or_none()
        if st_tok is None:
            self.session.logger.error("Clear DAQ restraints requires a Structure selected in Inputs.")
            return

        cmd = f"daq clearrestraints {st_tok}"
        self.session.logger.info(f"Running: {cmd}")
        run(self.session, cmd, log=False)
