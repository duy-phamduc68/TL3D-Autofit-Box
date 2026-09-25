"""
pick.py – Stage 0: Location picker.

Lets the user select a location folder, then choose whether to
Construct / Reconstruct / Validate the G_projection config.
"""
import os
from typing import Optional

from PyQt5.QtCore import Qt, QPointF, QSize
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QCheckBox, QDialog, QMessageBox,
    QDialogButtonBox, QGraphicsView, QGraphicsScene,
)
from PyQt5.QtSvg import QSvgRenderer

from ..config import default_config, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MediaViewer(QGraphicsView):
    """Minimal pan/zoom image+overlay viewer for the Pick stage preview."""

    PLACEHOLDER_SIZE = 18
    PLACEHOLDER_COLOR = QColor(120, 120, 120)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = None
        self._overlay_item = None
        self._placeholder_item = None
        self._last_image_path = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    # --- Loading ---------------------------------------------------------

    def load_image(self, path: str):
        self.scene().clear()
        self._pixmap_item = self._overlay_item = self._placeholder_item = None
        try:
            self.resetTransform()
        except Exception:
            pass
        pix = QPixmap(path)
        if pix and not pix.isNull():
            self._pixmap_item = self.scene().addPixmap(pix)
            self._last_image_path = path
            self.scene().setSceneRect(self._pixmap_item.boundingRect())
            self.fit_view()

    def set_placeholder(self, text: str):
        self.scene().clear()
        self._pixmap_item = self._overlay_item = self._last_image_path = None
        try:
            self.resetTransform()
        except Exception:
            pass
        font = QFont()
        font.setPointSize(self.PLACEHOLDER_SIZE)
        font.setBold(True)
        self._placeholder_item = self.scene().addText(text, font)
        try:
            self._placeholder_item.setDefaultTextColor(self.PLACEHOLDER_COLOR)
        except Exception:
            pass
        vw = max(400, self.viewport().width() or 400)
        vh = max(300, self.viewport().height() or 300)
        self.scene().setSceneRect(0, 0, vw, vh)
        r = self._placeholder_item.boundingRect()
        sr = self.scene().sceneRect()
        self._placeholder_item.setPos(
            QPointF((sr.width() - r.width()) / 2, (sr.height() - r.height()) / 2)
        )

    def load_svg(self, path: str):
        try:
            renderer = QSvgRenderer(path)
            if not renderer.isValid():
                return
            sz = self._pixmap_item.boundingRect().size().toSize() \
                if self._pixmap_item else renderer.defaultSize()
            if not sz or sz.width() == 0:
                sz = QSize(800, 600)
            img = QImage(sz, QImage.Format_ARGB32)
            img.fill(0)
            p = QPainter(img)
            renderer.render(p)
            p.end()
            pix = QPixmap.fromImage(img)
            self.scene().clear()
            try:
                self.resetTransform()
            except Exception:
                pass
            self._pixmap_item = self._overlay_item = self._placeholder_item = None
            self._pixmap_item = self.scene().addPixmap(pix)
            self.scene().setSceneRect(self._pixmap_item.boundingRect())
            self.fit_view()
        except Exception:
            pass

    def clear(self):
        self.scene().clear()
        try:
            self.resetTransform()
        except Exception:
            pass
        self._pixmap_item = self._overlay_item = self._placeholder_item = None
        self._last_image_path = None

    # --- Overlay ---------------------------------------------------------

    def set_overlay(self, pixmap: QPixmap):
        try:
            if self._overlay_item:
                if self._overlay_item.scene() == self.scene():
                    self.scene().removeItem(self._overlay_item)
                self._overlay_item = None
        except Exception:
            pass
        if pixmap and not pixmap.isNull():
            self._overlay_item = self.scene().addPixmap(pixmap)
            try:
                self._overlay_item.setZValue(1)
            except Exception:
                pass

    def clear_overlay(self):
        try:
            if self._overlay_item:
                if self._overlay_item.scene() == self.scene():
                    self.scene().removeItem(self._overlay_item)
                self._overlay_item = None
        except Exception:
            pass

    # --- Misc ------------------------------------------------------------

    def fit_view(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Stage widget
# ---------------------------------------------------------------------------

class PickStage(QWidget):
    """Stage 0 – pick a location and decide what to do with it."""

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root or os.getcwd()
        self.host_tab = None
        self.last_location: Optional[str] = None
        self.last_options: dict = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # --- Location row ------------------------------------------------
        loc_row = QHBoxLayout()
        loc_row.addWidget(QLabel("Location:"))
        self.combo = QComboBox()
        loc_row.addWidget(self.combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Refresh location list")
        self.refresh_btn.clicked.connect(self._on_refresh)
        loc_row.addWidget(self.refresh_btn)
        self._populate_location_list()
        self.combo.currentIndexChanged.connect(self._on_location_changed)
        layout.addLayout(loc_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Actions -----------------------------------------------------
        # Validation is the only action now

        # --- Media preview -----------------------------------------------
        media_box = QGroupBox("Media Preview")
        m_layout = QHBoxLayout(media_box)
        m_layout.setContentsMargins(6, 6, 6, 6)

        self.media1 = _MediaViewer()
        self.media1.setMinimumSize(320, 240)
        lv = QVBoxLayout()
        lv.addWidget(self.media1)
        self.media1_fit = QPushButton("Fit CCTV")
        self.media1_fit.clicked.connect(self.media1.fit_view)
        lv.addWidget(self.media1_fit)
        self.media1_roi_cb = QCheckBox("Show ROI overlay")
        self.media1_roi_cb.setEnabled(False)
        self.media1_roi_cb.toggled.connect(self._on_roi_toggled)
        lv.addWidget(self.media1_roi_cb)
        lw = QWidget(); lw.setLayout(lv)

        self.media2 = _MediaViewer()
        self.media2.setMinimumSize(320, 240)
        rv = QVBoxLayout()
        rv.addWidget(self.media2)
        self.media2_fit = QPushButton("Fit SAT")
        self.media2_fit.clicked.connect(self.media2.fit_view)
        rv.addWidget(self.media2_fit)
        self.media2_svg_cb = QCheckBox("Show SVG layout")
        self.media2_svg_cb.setEnabled(False)
        self.media2_svg_cb.toggled.connect(self._on_svg_toggled)
        rv.addWidget(self.media2_svg_cb)
        rw = QWidget(); rw.setLayout(rv)

        m_layout.addWidget(lw)
        m_layout.addWidget(rw)
        layout.addWidget(media_box, 1)

        # --- Summary / proceed -------------------------------------------
        self.summary = QLabel("No location selected.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.proceed_btn = QPushButton("Validate Location →")
        self.proceed_btn.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold; padding: 0 20px;")
        self.proceed_btn.setEnabled(False)
        self.proceed_btn.setFixedHeight(36)
        self.proceed_btn.clicked.connect(self._on_proceed)
        btn_row.addWidget(self.proceed_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _host(self):
        return getattr(self, "host_tab", None) or self.parent()

    def _populate_location_list(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("(none chosen)")
        loc_root = os.path.join(self.project_root, "location")
        try:
            dirs = sorted(
                n for n in os.listdir(loc_root)
                if os.path.isdir(os.path.join(loc_root, n))
            )
            for d in dirs:
                self.combo.addItem(d)
        except Exception:
            pass
        self.combo.blockSignals(False)

    def _loc_dir(self, code: str) -> Optional[str]:
        path = os.path.join(self.project_root, "location", code)
        return path if os.path.isdir(path) else None

    def _gproj_path(self, code: str) -> Optional[str]:
        d = self._loc_dir(code)
        if not d:
            return None
        p = os.path.join(d, f"G_projection_{code}.json")
        return p if os.path.isfile(p) else None

    def _check_required_files(self, code: str, use_svg: bool, use_roi: bool):
        missing = []
        d = self._loc_dir(code)
        if not d:
            return [f"location/{code} missing"]
        if not os.path.isfile(os.path.join(d, f"cctv_{code}.png")):
            missing.append(f"cctv_{code}.png")
        if not os.path.isfile(os.path.join(d, f"sat_{code}.png")):
            missing.append(f"sat_{code}.png")
        if use_svg and not os.path.isfile(os.path.join(d, f"layout_{code}.svg")):
            missing.append(f"layout_{code}.svg")
        if use_roi and not os.path.isfile(os.path.join(d, f"roi_{code}.png")):
            missing.append(f"roi_{code}.png")
        return missing

    def _warn_missing(self, missing):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Missing files")
        msg.setText("The following required files are missing:\n" + "\n".join(f"  • {m}" for m in missing))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def _refresh_timeline(self):
        host = self._host()
        if host and hasattr(host, "_update_progress_to_index") and hasattr(host, "current_step_index"):
            host._update_progress_to_index(host.current_step_index)

    def _update_summary(self):
        if not self.last_location:
            return
        act = self.last_options.get("action")
        lines = [f"Location: {self.last_location}"]
        if act:
            lines.append(f"Action: {act.upper()}")
        self.proceed_btn.setEnabled(bool(act))
        self.summary.setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_refresh(self):
        self._populate_location_list()
        self.combo.setCurrentIndex(0)
        self.status_label.setText("Location list refreshed.")

    def _on_location_changed(self, idx: int):
        if idx <= 0:
            self.last_location = None
            host = self._host()
            if host:
                host.inspect_obj = None
            self.status_label.setText("No location selected.")
            self.proceed_btn.setEnabled(False)
            self.last_options = {}
            self.media1.set_placeholder("No location loaded")
            self.media2.set_placeholder("No location loaded")
            self.media1.clear_overlay()
            self.media2.clear_overlay()
            self.media1_roi_cb.setChecked(False)
            self.media1_roi_cb.setEnabled(False)
            self.media2_svg_cb.setChecked(False)
            self.media2_svg_cb.setEnabled(False)
            self._update_summary()
            return
        self._load_location(self.combo.itemText(idx))

    def _load_location(self, code: str):
        self.last_location = code
        self.last_options = {}
        host = self._host()

        gpath = self._gproj_path(code)
        try:
            if gpath:
                cfg = load_config(gpath)
                if host:
                    host.inspect_obj = cfg
                self.status_label.setText(f"Loaded existing config: {gpath}")
                self.proceed_btn.setEnabled(True)
                self.last_options = {"action": "validate", "gproj_path": gpath}
            else:
                if host:
                    host.inspect_obj = None
                self.status_label.setText(f"No existing config for '{code}'. Validation requires an existing G_projection.")
                self.proceed_btn.setEnabled(False)
                self.last_options = {"action": "missing"}
            self._refresh_timeline()
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

        self._load_media_previews(code)
        self._update_summary()

    def _on_proceed(self):
        if not self.last_location:
            return
        action = self.last_options.get("action")
        if action != "validate":
            self.status_label.setText("Cannot proceed without a valid config.")
            return
        host = self._host()
        if host is None:
            return
        host._go_to_stage(1)

    # ------------------------------------------------------------------
    # Media preview helpers
    # ------------------------------------------------------------------

    def _load_media_previews(self, code: str):
        d = self._loc_dir(code)
        if not d:
            return
        host = self._host()
        cfg = getattr(host, "inspect_obj", None) or {}
        use_svg = bool(cfg.get("use_svg", False))
        use_roi = bool(cfg.get("use_roi", False))

        self.media1_roi_cb.setChecked(False)
        self.media1_roi_cb.setEnabled(False)
        self.media2_svg_cb.setChecked(False)
        self.media2_svg_cb.setEnabled(False)

        cctv = os.path.join(d, f"cctv_{code}.png")
        if os.path.isfile(cctv):
            self.media1.load_image(cctv)
        else:
            self.media1.set_placeholder("cctv image missing")

        sat = os.path.join(d, f"sat_{code}.png")
        if os.path.isfile(sat):
            self.media2.load_image(sat)
        else:
            self.media2.set_placeholder("sat image missing")

        roi = os.path.join(d, f"roi_{code}.png")
        if use_roi and os.path.isfile(roi):
            roi_pix = QPixmap(roi)
            if not roi_pix.isNull():
                overlay = QPixmap(roi_pix.size())
                overlay.fill(QColor(255, 0, 0, 100))
                mask = roi_pix.createMaskFromColor(QColor(0, 0, 0), Qt.MaskOutColor)
                overlay.setMask(mask)
                self._roi_overlay_pixmap = overlay
                self.media1_roi_cb.setEnabled(True)

        svg = os.path.join(d, f"layout_{code}.svg")
        if use_svg and os.path.isfile(svg):
            self._sat_svg_path = svg
            self.media2_svg_cb.setEnabled(True)

    def _on_roi_toggled(self, checked: bool):
        if checked and hasattr(self, "_roi_overlay_pixmap") and self._roi_overlay_pixmap:
            self.media1.set_overlay(self._roi_overlay_pixmap)
        else:
            self.media1.clear_overlay()

    def _on_svg_toggled(self, checked: bool):
        if checked and getattr(self, "_sat_svg_path", None):
            self.media2.load_svg(self._sat_svg_path)
        elif self.last_location:
            self._load_media_previews(self.last_location)
