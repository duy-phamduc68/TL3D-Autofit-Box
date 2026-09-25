import os
import math
import re
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QTransform, QFont, QPolygonF, QKeySequence
from PyQt5.QtSvg import QGraphicsSvgItem
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QCheckBox, QGroupBox, QGraphicsView, QGraphicsScene, 
    QGraphicsRectItem, QDoubleSpinBox, QComboBox, QSlider, 
    QScrollArea, QGraphicsPolygonItem, QSplitter, QLineEdit,
    QSpinBox, QShortcut
)

from ._base import ImageViewer, cv_to_qimage, remap_with_supersample

# ==========================================================
# CONSTANTS & CONFIGURATION
# ==========================================================
DEFAULT_BOX_L = 3.0
DEFAULT_BOX_W = 1.5
DEFAULT_BOX_H = 1.55

# ==========================================================
# ROBUST SVG PARSER (Matching g_projection.py)
# ==========================================================
class SVGParser:
    def __init__(self, svg_path, affine_matrix=None):
        self.svg_path = svg_path
        self.orientation_segments = []
        self.M_align = np.identity(3)
        if affine_matrix is not None:
            self.M_align[:2, :] = np.array(affine_matrix)
            
        self.valid = False
        if os.path.exists(svg_path):
            try:
                self.tree = ET.parse(svg_path)
                self.root = self.tree.getroot()
                self.orientation_segments = self._extract_segments()
                self.valid = True
            except Exception as e:
                print(f"[SVG ERR] {e}")
        else:
            print(f"[SVG ERR] File not found: {svg_path}")

    def _parse_transform(self, txt):
        M = np.identity(3)
        if not txt: return M
        ops = re.findall(r'(\w+)\s*\(([^)]+)\)', txt)
        for name, args in ops:
            vals = list(map(float, filter(None, re.split(r'[ ,]+', args.strip()))))
            T = np.identity(3)
            if name == 'translate':
                T[0,2], T[1,2] = vals[0], vals[1] if len(vals) > 1 else 0
            elif name == 'rotate':
                rad = math.radians(vals[0])
                c, s = math.cos(rad), math.sin(rad)
                if len(vals) == 3:
                    cx, cy = vals[1], vals[2]
                    T1=np.eye(3); T1[0,2]=cx; T1[1,2]=cy
                    R=np.eye(3); R[:2,:2]=[[c,-s],[s,c]]
                    T2=np.eye(3); T2[0,2]=-cx; T2[1,2]=-cy
                    T = T1 @ R @ T2
                else:
                    T[:2,:2] = [[c,-s],[s,c]]
            elif name == 'matrix':
                T = np.array([[vals[0], vals[2], vals[4]],
                              [vals[1], vals[3], vals[5]],
                              [0, 0, 1]])
            M = M @ T
        return M

    def _extract_segments(self):
        segs = []
        target_ids = ['Guidelines', 'Physical']
        
        def get_tag(el):
            return el.tag.split('}')[-1]

        target_nodes = []
        for el in self.root.iter():
            if get_tag(el) == 'g' and el.get('id') in target_ids:
                target_nodes.append(el)

        if not target_nodes:
            print(f"[SVG] Warning: No groups found with IDs {target_ids}")

        for g in target_nodes:
            self._process_node(g, np.identity(3), segs)
            
        return segs

    def _process_node(self, element, parent_mat, seg_list):
        local_mat = self._parse_transform(element.get('transform'))
        curr_mat = parent_mat @ local_mat
        
        tag = element.tag.split('}')[-1]
        pts = []
        
        if tag == 'line':
            pts = np.array([[float(element.get('x1',0)), float(element.get('y1',0))],
                            [float(element.get('x2',0)), float(element.get('y2',0))]])
        elif tag == 'polygon' or tag == 'polyline':
            raw = re.split(r'[ ,]+', element.get('points','').strip())
            raw = [x for x in raw if x]
            if raw: pts = np.array(raw, dtype=float).reshape(-1,2)
        
        if len(pts) > 0:
            homo = np.hstack([pts, np.ones((len(pts), 1))])
            t_pts = (self.M_align @ (curr_mat @ homo.T)).T[:, :2]
            
            for i in range(len(t_pts)-1):
                seg_list.append((t_pts[i], t_pts[i+1]))
            if tag == 'polygon':
                seg_list.append((t_pts[-1], t_pts[0]))

        for child in element:
            self._process_node(child, curr_mat, seg_list)

    def get_nearest_heading_info(self, pt):
        if not self.valid or not self.orientation_segments: return None, None, None
        min_d = float('inf')
        best_ang = None
        best_seg = None
        pt = np.array(pt)
        for sp1, sp2 in self.orientation_segments:
            ab = sp2 - sp1
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-6: continue
            ap = pt - sp1
            t = np.dot(ap, ab) / ab_sq
            closest = sp1 + np.clip(t, 0, 1) * ab
            d = np.linalg.norm(pt - closest)
            if d < min_d:
                min_d = d
                best_ang = math.degrees(math.atan2(ab[1], ab[0]))
                best_seg = (sp1, sp2)
        if best_ang is not None:
            return (best_ang + 360) % 360, best_seg[0], best_seg[1]
        return None, None, None

class BoxDrawViewer(ImageViewer):
    boxDrawn = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rect_item = None
        self._start_pos = None
        self._is_drawing = False
        self._overlay_item = None
        self._pen = QPen(Qt.yellow, 2, Qt.SolidLine)
        self._brush = QBrush(QColor(255, 255, 0, 50))
        self._crosshair_h = None
        self._crosshair_v = None
        self.setMouseTracking(True)

    def load_pixmap(self, pixmap: QPixmap):
        _scene = self.scene()
        for attr in ('_rect_item', '_overlay_item', '_crosshair_h', '_crosshair_v'):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    if item.scene() == _scene:
                        _scene.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)
        super().load_pixmap(pixmap)
        self._start_pos = None
        self._is_drawing = False

    def set_overlay(self, pixmap: QPixmap):
        if self._overlay_item:
            try:
                if self._overlay_item.scene() == self.scene(): self.scene().removeItem(self._overlay_item)
            except RuntimeError: pass
            self._overlay_item = None
        if pixmap and not pixmap.isNull():
            self._overlay_item = self.scene().addPixmap(pixmap)
            self._overlay_item.setZValue(1)

    def clear_overlay(self):
        if self._overlay_item:
            try:
                if self._overlay_item.scene() == self.scene(): self.scene().removeItem(self._overlay_item)
            except RuntimeError: pass
            self._overlay_item = None

    def mousePressEvent(self, event):
        if self._pixmap_item and event.button() == Qt.RightButton:
            self._start_pos = self.mapToScene(event.pos())
            self._is_drawing = True
            if self._rect_item:
                try:
                    if self._rect_item.scene() == self.scene(): self.scene().removeItem(self._rect_item)
                except RuntimeError: pass
                self._rect_item = None
            self._rect_item = QGraphicsRectItem()
            self._rect_item.setPen(self._pen)
            self._rect_item.setBrush(self._brush)
            self._rect_item.setZValue(10)
            self.scene().addItem(self._rect_item)
            event.accept()
        else:
            super().mousePressEvent(event)

    def leaveEvent(self, event):
        if getattr(self, '_crosshair_h', None): self._crosshair_h.hide()
        if getattr(self, '_crosshair_v', None): self._crosshair_v.hide()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        if self._pixmap_item:
            curr_pos = self.mapToScene(event.pos())
            brect = self._pixmap_item.boundingRect()
            
            if not getattr(self, '_crosshair_h', None):
                from PyQt5.QtWidgets import QGraphicsLineItem
                pen = QPen(QColor(255, 255, 255, 225), 1, Qt.DashLine)
                self._crosshair_h = QGraphicsLineItem()
                self._crosshair_v = QGraphicsLineItem()
                self._crosshair_h.setPen(pen)
                self._crosshair_v.setPen(pen)
                self._crosshair_h.setZValue(99)
                self._crosshair_v.setZValue(99)
                self.scene().addItem(self._crosshair_h)
                self.scene().addItem(self._crosshair_v)
            
            x, y = curr_pos.x(), curr_pos.y()
            if brect.contains(curr_pos):
                self._crosshair_h.setLine(brect.left(), y, brect.right(), y)
                self._crosshair_v.setLine(x, brect.top(), x, brect.bottom())
                self._crosshair_h.show()
                self._crosshair_v.show()
            else:
                self._crosshair_h.hide()
                self._crosshair_v.hide()

        if self._is_drawing and self._start_pos:
            curr_pos = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, curr_pos).normalized()
            brect = self._pixmap_item.boundingRect()
            clamped_rect = rect.intersected(brect)
            if self._rect_item: self._rect_item.setRect(clamped_rect)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._is_drawing:
            self._is_drawing = False
            if self._rect_item: self.boxDrawn.emit(self._rect_item.rect())
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class FinalStage(QWidget):
    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        self._K = None; self._D = None; self._new_K = None; self._H = None; self._H_inv = None
        self._z_cam = 10.0; self._cam_sat = np.zeros(2); self._px_per_m = 1.0
        
        self._svg_parser = None
        self._svg_affine = None
        self._mask_cv = None 
        
        self._current_rect = None 
        self._ref_point_cctv = None 
        self._proj_point_sat = None 
        self._gc_point_cctv = None  
        self._heading_deg = 0.0
        self._show_3d_active = True
        
        self._svg_item = None; self._roi_overlay = None
        self._sat_markers = []; self._cctv_markers = []; self._wireframe_items = []
        self._floor_poly = None; self._highlight_line = None 
        self._debug_items = []
        
        self.item_cctv_warped = None
        self._fov_poly_item = None
        self._cctv_files = []
        self._cctv_index = 0
        
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        sidebar = QWidget(); sidebar.setFixedWidth(320); sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_layout = QVBoxLayout(sidebar); side_layout.setSpacing(10)
        
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("border: none;")
        scroll_content = QWidget(); vbox = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        
        vbox.addWidget(QLabel("<h2>Final Validation</h2>"))
        
        grp0 = QGroupBox("1. Homography FOV"); l0 = QVBoxLayout(grp0)
        h0 = QHBoxLayout(); h0.addWidget(QLabel("Opacity:"))
        self.slider_fov_alpha = QSlider(Qt.Horizontal); self.slider_fov_alpha.setRange(0, 100); self.slider_fov_alpha.setValue(60)
        self.slider_fov_alpha.valueChanged.connect(self._on_fov_alpha_changed)
        h0.addWidget(self.slider_fov_alpha); l0.addLayout(h0)
        self.btn_compute_fov = QPushButton("Compute FOV & Warp")
        self.btn_compute_fov.setStyleSheet("background-color: #555; color: white;")
        self.btn_compute_fov.clicked.connect(self._on_compute_fov)
        l0.addWidget(self.btn_compute_fov)
        vbox.addWidget(grp0)
        
        grp1 = QGroupBox("2. Draw Box"); l1 = QVBoxLayout(grp1)
        l1.addWidget(QLabel("Right-click drag on CCTV."))
        self.btn_reset_box = QPushButton("Reset Box")
        self.btn_reset_box.setStyleSheet("border: 1px solid #555; padding: 4px; background-color: #444; color: #fff;")
        self.btn_reset_box.clicked.connect(self._on_reset_box)
        l1.addWidget(self.btn_reset_box)
        vbox.addWidget(grp1)
        
        grp2 = QGroupBox("3. Dimensions (m)"); l2 = QVBoxLayout(grp2)
        
        self.spin_w = self._make_spin(DEFAULT_BOX_W); self.spin_w.setReadOnly(True)
        self.spin_l = self._make_spin(DEFAULT_BOX_L); self.spin_l.setReadOnly(True)
        self.spin_h = self._make_spin(DEFAULT_BOX_H); self.spin_h.setReadOnly(True)
        
        h_dims1 = QHBoxLayout()
        h_dims1.addWidget(QLabel("L:")); h_dims1.addWidget(self.spin_l)
        h_dims1.addWidget(QLabel("W:")); h_dims1.addWidget(self.spin_w)
        l2.addLayout(h_dims1)
        
        h_dims2 = QHBoxLayout()
        h_dims2.addWidget(QLabel("H:")); h_dims2.addWidget(self.spin_h)
        l2.addLayout(h_dims2)
        
        v_weight = QVBoxLayout()
        def make_wt(val):
            s = QDoubleSpinBox(); s.setRange(0.01, 100000); s.setDecimals(1); s.setValue(val)
            return s
        self.spin_wt_l = make_wt(600.0)
        self.spin_wt_w = make_wt(200.0)
        self.spin_wt_h = make_wt(400.0)
        
        h_wt1 = QHBoxLayout()
        h_wt1.addWidget(QLabel("Wt L:")); h_wt1.addWidget(self.spin_wt_l)
        h_wt1.addWidget(QLabel("Wt W:")); h_wt1.addWidget(self.spin_wt_w)
        v_weight.addLayout(h_wt1)
        
        h_wt2 = QHBoxLayout()
        h_wt2.addWidget(QLabel("Wt H:")); h_wt2.addWidget(self.spin_wt_h)
        v_weight.addLayout(h_wt2)
        
        l2.addLayout(v_weight)
        
        h_inf = QHBoxLayout()
        h_inf.addWidget(QLabel("Inflate V:"))
        self.slider_inf_v = QSlider(Qt.Horizontal); self.slider_inf_v.setRange(100, 200); self.slider_inf_v.setValue(100)
        self.slider_inf_v.valueChanged.connect(self._refresh_visuals)
        h_inf.addWidget(self.slider_inf_v)
        h_inf.addWidget(QLabel("Inflate H:"))
        self.slider_inf_h = QSlider(Qt.Horizontal); self.slider_inf_h.setRange(100, 200); self.slider_inf_h.setValue(100)
        self.slider_inf_h.valueChanged.connect(self._refresh_visuals)
        h_inf.addWidget(self.slider_inf_h)
        l2.addLayout(h_inf)
        
        self.chk_show_inflated = QCheckBox("Show inflated box")
        self.chk_show_inflated.setChecked(False)
        self.chk_show_inflated.toggled.connect(self._refresh_visuals)
        l2.addWidget(self.chk_show_inflated)
        
        h_opt1 = QHBoxLayout()
        self.btn_run_opt = QPushButton("Run Optimization (O)")
        self.btn_run_opt.setStyleSheet("background-color: #d88; font-weight: bold;")
        self.btn_run_opt.clicked.connect(self._run_optimization)
        h_opt1.addWidget(self.btn_run_opt)
        
        h_opt2 = QHBoxLayout()
        self.chk_force_touch = QCheckBox("Force touch all sides")
        self.chk_force_touch.setChecked(True)
        h_opt2.addWidget(self.chk_force_touch)
        
        h_opt2.addWidget(QLabel("Iters:"))
        self.spin_iters = QSpinBox(); self.spin_iters.setRange(10, 5000); self.spin_iters.setValue(750)
        h_opt2.addWidget(self.spin_iters)
        
        l2.addLayout(h_opt1)
        l2.addLayout(h_opt2)

        vbox.addWidget(grp2)
        
        QShortcut(QKeySequence("O"), self).activated.connect(self._run_optimization)
        
        self._shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._shortcut_left.activated.connect(self._on_prev_cctv)
        self._shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._shortcut_right.activated.connect(self._on_next_cctv)
        
        self._shortcut_a = QShortcut(QKeySequence("A"), self)
        self._shortcut_a.activated.connect(self._on_heading_step_down)
        self._shortcut_d = QShortcut(QKeySequence("D"), self)
        self._shortcut_d.activated.connect(self._on_heading_step_up)
        
        self._shortcut_shift_a = QShortcut(QKeySequence("Shift+A"), self)
        self._shortcut_shift_a.activated.connect(self._on_heading_step_down_90)
        self._shortcut_shift_d = QShortcut(QKeySequence("Shift+D"), self)
        self._shortcut_shift_d.activated.connect(self._on_heading_step_up_90)
        
        grp3 = QGroupBox("4. Projection"); l3 = QVBoxLayout(grp3)
        self.btn_toggle_pts = QPushButton("Toggle Points")
        self.btn_toggle_pts.setCheckable(True)
        self.btn_toggle_pts.setChecked(True)
        self.btn_toggle_pts.setStyleSheet("QPushButton { background-color: #2a84ff; color: white; font-weight: bold; border: 1px solid #1a64db; padding: 4px; } QPushButton:checked { background-color: #1a64db; }")
        self.btn_toggle_pts.toggled.connect(self._on_toggle_points)
        l3.addWidget(self.btn_toggle_pts)
        self.chk_show_floor_box = QCheckBox("Show default floor box")
        self.chk_show_floor_box.setChecked(True)
        self.chk_show_floor_box.toggled.connect(self._on_toggle_floor_box)
        l3.addWidget(self.chk_show_floor_box)
        vbox.addWidget(grp3)
        
        grp4 = QGroupBox("5. Heading & Floor"); l4 = QVBoxLayout(grp4)
        self.chk_auto_head = QCheckBox("Auto Heading (SVG)"); self.chk_auto_head.setChecked(False)
        self.chk_auto_head.toggled.connect(self._toggle_heading_mode)
        l4.addWidget(self.chk_auto_head)
        h4 = QHBoxLayout(); h4.addWidget(QLabel("Angle:"))
        self.slider_head = QSlider(Qt.Horizontal); self.slider_head.setRange(0, 360)
        self.slider_head.setEnabled(True)
        self.slider_head.valueChanged.connect(self._on_heading_changed)
        h4.addWidget(self.slider_head)
        l4.addLayout(h4); vbox.addWidget(grp4)
        
        grp5 = QGroupBox("6. 3D Reconstruction"); l5 = QVBoxLayout(grp5)
        self.btn_show_3d = QPushButton("Toggle 3D Box"); self.btn_show_3d.setCheckable(True)
        self.btn_show_3d.setChecked(True)
        self.btn_show_3d.setStyleSheet("QPushButton { background-color: #2ca02c; color: white; font-weight: bold; border: 1px solid #1c801c; padding: 4px; } QPushButton:checked { background-color: #1e701e; }")
        self.btn_show_3d.toggled.connect(self._on_toggle_3d)
        l5.addWidget(self.btn_show_3d)
        vbox.addWidget(grp5)
        
        grp_opt = QGroupBox("Options"); l_opt = QVBoxLayout(grp_opt)
        self.chk_roi = QCheckBox("Show ROI Mask"); self.chk_roi.toggled.connect(self._on_toggle_roi)
        l_opt.addWidget(self.chk_roi)
        h_svg = QHBoxLayout(); h_svg.addWidget(QLabel("SVG Alpha:"))
        self.slider_alpha = QSlider(Qt.Horizontal); self.slider_alpha.setRange(0, 100); self.slider_alpha.setValue(50)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        h_svg.addWidget(self.slider_alpha)
        l_opt.addLayout(h_svg)
        vbox.addWidget(grp_opt)
        
        self.lbl_status = QLabel("Ready"); self.lbl_status.setWordWrap(True); self.lbl_status.setStyleSheet("color: #aaa; font-style: italic;")
        vbox.addWidget(self.lbl_status)
        
        self.btn_proceed = QPushButton("Save Config"); self.btn_proceed.setFixedHeight(40)
        self.btn_proceed.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; font-size: 14px; border: 1px solid #a04000; border-radius: 4px;")
        self.btn_proceed.clicked.connect(self._on_save_config)
        vbox.addWidget(self.btn_proceed)
        
        side_layout.addWidget(scroll)
        main_layout.addWidget(sidebar)
        
        splitter = QSplitter(Qt.Horizontal)
        left_c = QWidget(); lv = QVBoxLayout(left_c); lv.setContentsMargins(0,0,0,0)
        lv.addWidget(QLabel("CCTV (Right-Click Drag)"))
        self.view_cctv = BoxDrawViewer(); self.view_cctv.boxDrawn.connect(self._on_box_drawn)
        lv.addWidget(self.view_cctv)
        right_c = QWidget(); rv = QVBoxLayout(right_c); rv.setContentsMargins(0,0,0,0)
        rv.addWidget(QLabel("Satellite / SVG"))
        self.view_sat = ImageViewer()
        rv.addWidget(self.view_sat)
        splitter.addWidget(left_c); splitter.addWidget(right_c)
        main_layout.addWidget(splitter, 1)

    def _make_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.1, 50.0); s.setSingleStep(0.1); s.setValue(val)
        return s

    def _on_fov_alpha_changed(self, val):
        opacity = val / 100.0
        if self.item_cctv_warped: self.item_cctv_warped.setOpacity(opacity)
        if self._fov_poly_item: self._fov_poly_item.setOpacity(opacity)



    def _on_compute_fov(self):
        if not HAS_CV2: return
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        obj = host.inspect_obj
        
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        if getattr(self, '_cctv_files', None) and len(self._cctv_files) > 0:
            cctv_path = self._cctv_files[self._cctv_index]
        else:
            cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        sat_path  = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        
        if not (os.path.exists(cctv_path) and os.path.exists(sat_path)):
            self.lbl_status.setText("Missing images for FOV.")
            return

        img_cctv = cv2.imread(cctv_path)
        img_sat  = cv2.imread(sat_path)
        K = self._K if self._K is not None else np.eye(3)
        D = self._D if self._D is not None else np.zeros(5)
        
        img_cctv_undist = remap_with_supersample(img_cctv, K, D, K.copy())
        if self._H is None:
            self.lbl_status.setText("Missing Homography matrix.")
            return
            
        h_s, w_s = img_sat.shape[:2]
        px, py = w_s, h_s
        cw, ch = w_s + 2 * px, h_s + 2 * py
        T = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=np.float32)
        H_final = T @ self._H

        img_cctv_bgra = cv2.cvtColor(img_cctv_undist, cv2.COLOR_BGR2BGRA)
        warped = cv2.warpPerspective(img_cctv_bgra, H_final, (cw, ch))
        rgba = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
        
        h2, w2, ch2 = rgba.shape
        q_cctv = QImage(rgba.data.tobytes(), w2, h2, ch2 * w2, QImage.Format_RGBA8888).copy()
        
        if self.item_cctv_warped:
            try:
                if self.item_cctv_warped.scene() == self.view_sat.scene():
                    self.view_sat.scene().removeItem(self.item_cctv_warped)
            except Exception: pass
            
        from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsPolygonItem
        self.item_cctv_warped = QGraphicsPixmapItem(QPixmap.fromImage(q_cctv))
        self.item_cctv_warped.setPos(-px, -py)
        self.item_cctv_warped.setOpacity(self.slider_fov_alpha.value() / 100.0)
        self.item_cctv_warped.setZValue(0.5)
        self.view_sat.scene().addItem(self.item_cctv_warped)

        h_u, w_u = img_cctv.shape[:2]
        mask_src = 255 * np.ones((h_u, w_u), dtype=np.uint8)
        mask_warped = cv2.warpPerspective(mask_src, H_final, (cw, ch))
        contours, _ = cv2.findContours(mask_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self._fov_poly_item:
            try:
                if self._fov_poly_item.scene() == self.view_sat.scene():
                    self.view_sat.scene().removeItem(self._fov_poly_item)
            except Exception: pass

        if contours:
            fov = max(contours, key=cv2.contourArea)
            fov = cv2.approxPolyDP(fov, 0.001 * cv2.arcLength(fov, True), True)
            pts = []
            fov_sat_coords = []
            for p in fov[:, 0, :]:
                sx, sy = float(p[0]) - px, float(p[1]) - py
                pts.append(QPointF(sx, sy))
                fov_sat_coords.append([sx, sy])
            if pts: pts.append(pts[0])
            
            poly_item = QGraphicsPolygonItem(QPolygonF(pts))
            pen_g = QPen(Qt.green); pen_g.setWidth(3)
            poly_item.setPen(pen_g); poly_item.setBrush(QBrush(Qt.NoBrush))
            poly_item.setZValue(0.6)
            poly_item.setOpacity(self.slider_fov_alpha.value() / 100.0)
            self.view_sat.scene().addItem(poly_item)
            self._fov_poly_item = poly_item
            
            if "homography" not in obj: obj["homography"] = {}
            obj["homography"]["fov_polygon"] = fov_sat_coords
            self.lbl_status.setText("FOV computed and projected.")
        
        self.view_sat.scene().setSceneRect(-px, -py, cw, ch)
        self.view_sat.fitInView(self.view_sat.scene().sceneRect(), Qt.KeepAspectRatio)

    def _full_scene_reset(self):
        sat_scene = self.view_sat.scene() if self.view_sat else None
        cctv_scene = self.view_cctv.scene() if self.view_cctv else None

        def _remove(scene, item):
            if item is None or scene is None:
                return
            try:
                if item.scene() == scene:
                    scene.removeItem(item)
            except Exception:
                pass

        _remove(sat_scene, self._svg_item)
        self._svg_item = None
        _remove(sat_scene, self._floor_poly)
        self._floor_poly = None
        _remove(sat_scene, self._highlight_line)
        self._highlight_line = None
        _remove(sat_scene, getattr(self, 'item_cctv_warped', None))
        self.item_cctv_warped = None
        _remove(sat_scene, getattr(self, '_fov_poly_item', None))
        self._fov_poly_item = None
        _remove(sat_scene, getattr(self, '_fov_box_footprint_item', None))
        self._fov_box_footprint_item = None
        for it in getattr(self, '_debug_items', []):
            _remove(sat_scene, it)
        self._debug_items = []
        for it in self._sat_markers:
            _remove(sat_scene, it)
        self._sat_markers = []

        for it in self._wireframe_items:
            _remove(cctv_scene, it)
        self._wireframe_items = []
        for it in self._cctv_markers:
            _remove(cctv_scene, it)
        self._cctv_markers = []
        _remove(cctv_scene, getattr(self.view_cctv, '_overlay_item', None))
        self.view_cctv._overlay_item = None
        _remove(cctv_scene, getattr(self.view_cctv, '_rect_item', None))
        self.view_cctv._rect_item = None

        _remove(sat_scene, getattr(self, '_trapezoid_poly_item', None))
        self._trapezoid_poly_item = None
        
        self._roi_overlay = None
        self._floor_poly = None
        self._show_3d_active = self.btn_show_3d.isChecked() if hasattr(self, 'btn_show_3d') else True
        self._current_rect = None
        self._ref_point_cctv = None
        self._proj_point_sat = None
        self._gc_point_cctv = None
        self._heading_deg = 0.0
        if hasattr(self, '_floor_corners_sat'):
            del self._floor_corners_sat
        if hasattr(self, '_trapezoid_sat_corners'):
            del self._trapezoid_sat_corners

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2: return
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return

        self._full_scene_reset()

        try:
            self._load_params(host.inspect_obj)
            self._load_images(host.inspect_obj)
            self._init_svg_parser(host.inspect_obj)
        except Exception as e:
            self.lbl_status.setText(f"Error loading: {e}")

    def _load_params(self, obj):
        und = obj.get('undistort', {})
        K = und.get('K'); D = und.get('D', [0]*5)
        self._K = np.array(K, dtype=np.float64) if K else None
        self._D = np.array(D, dtype=np.float64)
        
        hom = obj.get('homography', {})
        H = hom.get('H')
        if H:
            self._H = np.array(H, dtype=np.float64)
            self._H_inv = np.linalg.inv(self._H)
            
        par = obj.get('parallax', {})
        self._z_cam = par.get('z_cam_meters', 10.0)
        self._cam_sat = np.array([par.get('x_cam_coords_sat',0), par.get('y_cam_coords_sat',0)])
        self._px_per_m = par.get('px_per_meter', 1.0)
        if self._px_per_m <= 0.001: self._px_per_m = 10.0
        
        layout = obj.get('layout_svg', {})
        A = layout.get('A') 
        if A: self._svg_affine = np.array(A)
        
        use_svg = obj.get('use_svg', False)
        self.chk_auto_head.setEnabled(use_svg)
        self.slider_alpha.setEnabled(use_svg)
        if not use_svg: self.chk_auto_head.setChecked(False)
        try:
            self.chk_roi.setEnabled(use_svg)
            if not use_svg:
                try:
                    self.chk_roi.blockSignals(True)
                    self.chk_roi.setChecked(False)
                finally:
                    try: self.chk_roi.blockSignals(False)
                    except: pass
                try:
                    if hasattr(self, 'view_cctv'):
                        self.view_cctv.clear_overlay()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_prev_cctv(self):
        if not getattr(self, '_cctv_files', None) or len(self._cctv_files) <= 1:
            return
        self._cctv_index = (self._cctv_index - 1) % len(self._cctv_files)
        self._load_cctv_image()

    def _on_next_cctv(self):
        if not getattr(self, '_cctv_files', None) or len(self._cctv_files) <= 1:
            return
        self._cctv_index = (self._cctv_index + 1) % len(self._cctv_files)
        self._load_cctv_image()

    def _load_cctv_image(self):
        if not self._cctv_files:
            return
        cctv_path = self._cctv_files[self._cctv_index]
        img = cv2.imread(cctv_path)
        if img is not None:
            if self._K is not None:
                self._new_K = self._K.copy()
            self.view_cctv.change_background_pixmap(QPixmap.fromImage(cv_to_qimage(img)))
            self.lbl_status.setText(f"CCTV: {os.path.basename(cctv_path)} ({self._cctv_index+1}/{len(self._cctv_files)})")

    def _on_heading_step_down(self):
        val = self.slider_head.value()
        new_val = (val - 5) % 360
        self.slider_head.setValue(new_val)

    def _on_heading_step_up(self):
        val = self.slider_head.value()
        new_val = (val + 5) % 360
        self.slider_head.setValue(new_val)

    def _on_heading_step_down_90(self):
        val = self.slider_head.value()
        new_val = (val - 90) % 360
        self.slider_head.setValue(new_val)

    def _on_heading_step_up_90(self):
        val = self.slider_head.value()
        new_val = (val + 90) % 360
        self.slider_head.setValue(new_val)

    def _load_images(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        
        loc_dir = os.path.join(proj_root, 'location', loc_code)
        cctv_files = []
        if os.path.isdir(loc_dir):
            for f in os.listdir(loc_dir):
                if f.lower().endswith(f"cctv_{loc_code.lower()}.png"):
                    cctv_files.append(os.path.join(loc_dir, f))
        
        def cctv_sort_key(filepath):
            filename = os.path.basename(filepath).lower()
            if filename == f"cctv_{loc_code.lower()}.png":
                return (-1, filename)
            prefix = filename.split('_')[0]
            try:
                return (0, int(prefix))
            except ValueError:
                return (1, filename)
                
        cctv_files.sort(key=cctv_sort_key)
        self._cctv_files = cctv_files
        self._cctv_index = 0
        
        if self._cctv_files:
            cctv_path = self._cctv_files[0]
            img = cv2.imread(cctv_path)
            if img is not None:
                if self._K is not None: self._new_K = self._K.copy()
                self.view_cctv.load_pixmap(QPixmap.fromImage(cv_to_qimage(img)))
                self.view_cctv.fitToView()
                self.lbl_status.setText(f"CCTV: {os.path.basename(cctv_path)} (1/{len(self._cctv_files)})")
                
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            img = cv2.imread(sat_path)
            if img is not None:
                self.view_sat.load_pixmap(QPixmap.fromImage(cv_to_qimage(img)))
                self.view_sat.fitToView()
        
        if obj.get('use_svg'):
            svg_path = os.path.join(proj_root, 'location', loc_code, f'layout_{loc_code}.svg')
            if os.path.isfile(svg_path):
                self._svg_item = QGraphicsSvgItem(svg_path)
                if self._svg_affine is not None:
                    m = self._svg_affine
                    trans = QTransform(m[0,0], m[1,0], m[0,1], m[1,1], m[0,2], m[1,2])
                    self._svg_item.setTransform(trans)
                self._svg_item.setOpacity(self.slider_alpha.value() / 100.0)
                self.view_sat.scene().addItem(self._svg_item)

        self._mask_cv = None
        self._roi_overlay = None 
        roi_path = os.path.join(proj_root, 'location', loc_code, f'roi_{loc_code}.png')
        
        if os.path.isfile(roi_path):
            raw_img = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)
            if raw_img is not None:
                h, w = raw_img.shape[:2]
                if raw_img.ndim == 3 and raw_img.shape[2] == 4:
                    alpha = raw_img[:, :, 3]
                    self._mask_cv = cv2.bitwise_not(alpha)
                elif raw_img.ndim == 3:
                    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                    _, self._mask_cv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                else:
                    self._mask_cv = raw_img

                arr = np.zeros((h, w, 4), dtype=np.uint8)
                is_invalid = (self._mask_cv < 10)
                arr[is_invalid, 0] = 0    
                arr[is_invalid, 1] = 0    
                arr[is_invalid, 2] = 255  
                arr[is_invalid, 3] = 60   
                
                img_data = QImage(arr.data, w, h, w*4, QImage.Format_ARGB32).copy()
                self._roi_overlay = QPixmap.fromImage(img_data)
                
                if self.chk_roi.isChecked():
                    self.view_cctv.set_overlay(self._roi_overlay)

    def _init_svg_parser(self, obj):
        if not obj.get('use_svg'): 
            self._svg_parser = None
            return
            
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        svg_path = os.path.join(proj_root, 'location', loc_code, f'layout_{loc_code}.svg')
        matrix_a = obj.get('layout_svg', {}).get('A')
        
        self._svg_parser = SVGParser(svg_path, matrix_a)
        if not self._svg_parser.valid:
            self.lbl_status.setText("Warning: SVG file parsed but no segments found.")
        elif not self._svg_parser.orientation_segments:
            self.lbl_status.setText("Warning: No Guidelines/Physical lines found in SVG.")
        else:
            self.lbl_status.setText("Auto Heading enabled.")

    def _on_box_drawn(self, rect):
        self._current_rect = rect
        self._clear_markers() 
        if self._mask_cv is not None:
            host = getattr(self, 'host_tab', None) or self.parent()
            roi_method = 'partial'
            if getattr(host, 'inspect_obj', None):
                roi_method = host.inspect_obj.get('roi_method', 'partial')
            valid = self._check_roi(rect, roi_method)
            if not valid:
                self.lbl_status.setText(f"Box rejected by ROI ({roi_method})")
                if self.view_cctv._rect_item: self.view_cctv._rect_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
                return
            else:
                if self.view_cctv._rect_item: self.view_cctv._rect_item.setPen(QPen(Qt.green, 2, Qt.DashLine))
        self.lbl_status.setText(f"Box drawn: {int(rect.width())}x{int(rect.height())}")
        self._on_confirm_points()

    def _check_roi(self, rect, method):
        if self._mask_cv is None: return True
        mask = self._mask_cv
        h_img, w_img = mask.shape[:2]

        x = max(0, min(w_img-1, int(rect.x())))
        y = max(0, min(h_img-1, int(rect.y())))
        w = int(rect.width())
        h = int(rect.height())

        corners = [
            (x, y),
            (min(x+w, w_img-1), y),
            (x, min(y+h, h_img-1)),
            (min(x+w, w_img-1), min(y+h, h_img-1))
        ]

        valid_corners = 0
        for cx, cy in corners:
            if mask[cy, cx] > 128:
                valid_corners += 1

        if method == 'in': return valid_corners == 4
        return valid_corners >= 1

    def _on_reset_box(self):
        self._current_rect = None
        if self.view_cctv._rect_item:
            try:
                if self.view_cctv._rect_item.scene() == self.view_cctv.scene():
                    self.view_cctv.scene().removeItem(self.view_cctv._rect_item)
            except RuntimeError: pass
            self.view_cctv._rect_item = None
        self._clear_markers()

    def _fit_3d_box(self):
        if not hasattr(self, 'slider_inf_v'): return
        inf_v = self.slider_inf_v.value() / 100.0
        inf_h = self.slider_inf_h.value() / 100.0
        rect = self._current_rect
        if not rect: return
        cx_2d = rect.x() + rect.width()/2
        cy_2d = rect.y() + rect.height()/2
        nw = rect.width() * inf_h
        nh = rect.height() * inf_v
        inf_rect = QRectF(cx_2d - nw/2, cy_2d - nh/2, nw, nh)
        
        cur_l = DEFAULT_BOX_L
        cur_w = DEFAULT_BOX_W
        cur_h = DEFAULT_BOX_H
        if self._proj_point_sat is None: return
        
        pt_app = np.array(self._proj_point_sat)
        vec_app = pt_app - self._cam_sat
        h_centroid = cur_h / 2.0
        if self._z_cam != h_centroid and self._z_cam != 0:
            factor = self._z_cam / (self._z_cam - h_centroid)
        else:
            factor = 1.0
        pt_true = self._cam_sat + (vec_app / factor)
        cur_cx, cur_cy = pt_true[0], pt_true[1]
        
        def get_penalty(l, w, h, cx, cy, force_touch=False):
            dx, dy = l/2.0 * self._px_per_m, w/2.0 * self._px_per_m
            corners = [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]
            rad = math.radians(self._heading_deg)
            c, s = math.cos(rad), math.sin(rad)
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            for x, y in corners:
                rx = x*c - y*s + cx; ry = x*s + y*c + cy
                pt_true_floor = np.array([rx, ry])
                
                # Floor at Z=0
                u, v = self._sat_to_cctv(pt_true_floor)
                min_x = min(min_x, u); max_x = max(max_x, u)
                min_y = min(min_y, v); max_y = max(max_y, v)
                
                # Ceil at Z=h
                vec = pt_true_floor - self._cam_sat
                if self._z_cam != h and self._z_cam != 0: factor = self._z_cam / (self._z_cam - h)
                else: factor = 100.0
                pt_ceil = self._cam_sat + (vec * factor)
                u, v = self._sat_to_cctv(pt_ceil)
                min_x = min(min_x, u); max_x = max(max_x, u)
                min_y = min(min_y, v); max_y = max(max_y, v)
                
            pen = 0.0
            if force_touch:
                pen += (inf_rect.left() - min_x)**2
                pen += (inf_rect.top() - min_y)**2
                pen += (max_x - inf_rect.right())**2
                pen += (max_y - inf_rect.bottom())**2
            else:
                if min_x < inf_rect.left(): pen += (inf_rect.left() - min_x)**2
                if min_y < inf_rect.top(): pen += (inf_rect.top() - min_y)**2
                if max_x > inf_rect.right(): pen += (max_x - inf_rect.right())**2
                if max_y > inf_rect.bottom(): pen += (max_y - inf_rect.bottom())**2
            return pen

        # Phase 1: Ensure it fits initially
        for _ in range(100):
            if get_penalty(cur_l, cur_w, cur_h, cur_cx, cur_cy, False) == 0:
                break
            cur_l *= 0.95; cur_w *= 0.95; cur_h *= 0.95

        weights = {
            'L': getattr(self, 'spin_wt_l', None).value() if hasattr(self, 'spin_wt_l') else 1000.0,
            'W': getattr(self, 'spin_wt_w', None).value() if hasattr(self, 'spin_wt_w') else 0.1,
            'H': getattr(self, 'spin_wt_h', None).value() if hasattr(self, 'spin_wt_h') else 10.0,
        }

        is_force_touch = getattr(self, 'chk_force_touch', None) and self.chk_force_touch.isChecked()

        def get_loss(p):
            l, w, h, cx, cy = p
            pen = get_penalty(l, w, h, cx, cy, is_force_touch)
            score = l * weights['L'] + h * weights['H'] + w * weights['W']
            ratio_pen = (h - 1.2 * w)**2
            return -score + 100000.0 * pen + 10000.0 * ratio_pen

        params = np.array([cur_l, cur_w, cur_h, cur_cx, cur_cy], dtype=np.float64)
        m_opt = np.zeros(5)
        v_opt = np.zeros(5)
        eps = 1e-4
        
        w_max = max(weights['L'], weights['W'], weights['H'], 1e-5)
        lr = np.array([
            0.05 * (weights['L'] / w_max),
            0.05 * (weights['W'] / w_max),
            0.05 * (weights['H'] / w_max),
            0.5 * self._px_per_m, 
            0.5 * self._px_per_m
        ])
        
        best_valid_params = params.copy()
        best_loss = float('inf')

        beta1 = 0.9
        beta2 = 0.999
        
        max_iters = getattr(self, 'spin_iters', None)
        iters = max_iters.value() if max_iters else 300
        
        for i in range(1, iters + 1):
            loss = get_loss(params)
            
            if loss < best_loss:
                best_loss = loss
                best_valid_params = params.copy()

            grad = np.zeros(5)
            for j in range(5):
                p_plus = params.copy()
                p_plus[j] += eps
                loss_plus = get_loss(p_plus)
                grad[j] = (loss_plus - loss) / eps
                
            m_opt = beta1 * m_opt + (1 - beta1) * grad
            v_opt = beta2 * v_opt + (1 - beta2) * (grad ** 2)
            m_hat = m_opt / (1 - beta1 ** i)
            v_hat = v_opt / (1 - beta2 ** i)
            
            params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            params[0] = max(DEFAULT_BOX_L, params[0])
            params[1] = max(DEFAULT_BOX_W, params[1])
            params[2] = max(DEFAULT_BOX_H, params[2])

        self._fitted_l, self._fitted_w, self._fitted_h, self._fitted_cx, self._fitted_cy = best_valid_params
        self.spin_l.setValue(self._fitted_l)
        self.spin_w.setValue(self._fitted_w)
        self.spin_h.setValue(self._fitted_h)

    def _update_inflated_box_visual(self):
        if not hasattr(self, 'slider_inf_v') or not self._current_rect: return
        inf_v = self.slider_inf_v.value() / 100.0
        inf_h = self.slider_inf_h.value() / 100.0
        rect = self._current_rect
        cx = rect.x() + rect.width()/2
        cy = rect.y() + rect.height()/2
        nw = rect.width() * inf_h
        nh = rect.height() * inf_v
        inf_rect = QRectF(cx - nw/2, cy - nh/2, nw, nh)
        
        if self.chk_show_inflated.isChecked():
            if getattr(self.view_cctv, '_rect_item', None):
                self.view_cctv._rect_item.setRect(inf_rect)
        else:
            if getattr(self.view_cctv, '_rect_item', None):
                self.view_cctv._rect_item.setRect(self._current_rect)

    def _refresh_visuals(self):
        if self._current_rect is not None:
            self._update_inflated_box_visual()
        if self._proj_point_sat is not None:
            self._draw_floor_box()
            if self._show_3d_active: self._on_show_3d()

    def _run_optimization(self):
        if self._proj_point_sat is not None:
            self._fit_3d_box()
            self._draw_floor_box()
            if self._show_3d_active: self._on_show_3d()

    def _fit_init_box_to_trapezoid(self):
        if not hasattr(self, '_trapezoid_sat_corners'): return
        import itertools
        
        trapezoid_pts = self._trapezoid_sat_corners
        rad = math.radians(-self._heading_deg)
        c, s = math.cos(rad), math.sin(rad)
        
        # Rotate trapezoid so the target box is axis-aligned
        R = np.array([[c, -s], [s, c]])
        rot_pts = np.dot(trapezoid_pts, R.T)
        
        edges = []
        for i in range(4):
            p1 = rot_pts[i]
            p2 = rot_pts[(i+1)%4]
            A = p2[1] - p1[1]
            B = -(p2[0] - p1[0])
            C_val = A * p1[0] + B * p1[1]
            edges.append({'A': A, 'B': B, 'C': C_val, 'p1': p1, 'p2': p2})
            
        best_area = -1
        best_params = None
        
        # Determine inside direction for polygon to check validity
        signed_area = 0.5 * np.sum(rot_pts[:,0]*np.roll(rot_pts[:,1], -1) - rot_pts[:,1]*np.roll(rot_pts[:,0], -1))
        is_ccw = signed_area > 0

        for perm in itertools.permutations(edges):
            E = perm
            M = np.array([
                [E[0]['A'], E[0]['B'], 0, 0],
                [0, E[1]['B'], E[1]['A'], 0],
                [0, 0, E[2]['A'], E[2]['B']],
                [E[3]['A'], 0, 0, E[3]['B']]
            ])
            try:
                v = np.linalg.solve(M, [E[0]['C'], E[1]['C'], E[2]['C'], E[3]['C']])
            except np.linalg.LinAlgError:
                continue
                
            xmin, ymin, xmax, ymax = v
            if xmin >= xmax or ymin >= ymax:
                continue
                
            corners = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]
            ])
            
            # Check if all corners are inside or on the boundaries of the polygon
            valid = True
            for cx, cy in corners:
                for edge in edges:
                    A, B, C_val = edge['A'], edge['B'], edge['C']
                    # For CCW, Ax + By <= C_val is inside
                    val = A*cx + B*cy
                    if is_ccw:
                        if val > C_val + 1e-3: valid = False
                    else:
                        if val < C_val - 1e-3: valid = False
                if not valid: break
                
            if valid:
                area = (xmax - xmin) * (ymax - ymin)
                if area > best_area:
                    best_area = area
                    
                    cx_rot = (xmin + xmax) / 2
                    cy_rot = (ymin + ymax) / 2
                    
                    # Rotate center back to original orientation
                    R_inv = np.array([[c, s], [-s, c]])
                    center_orig = np.dot([cx_rot, cy_rot], R_inv.T)
                    
                    best_params = (center_orig[0], center_orig[1], xmax - xmin, ymax - ymin)
                    
        if best_params:
            self._fitted_cx = best_params[0]
            self._fitted_cy = best_params[1]
            self._fitted_l = best_params[2] / self._px_per_m
            self._fitted_w = best_params[3] / self._px_per_m
            
            # Update UI
            self.spin_l.setValue(self._fitted_l)
            self.spin_w.setValue(self._fitted_w)
        else:
            # Fallback if mathematically no fully inscribed matching all 4 sides exists
            self.lbl_status.setText("Could not find analytical box touching all 4 sides.")

    def _on_confirm_points(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and getattr(host, 'inspect_obj', None):
            host.inspect_obj['ref_method'] = 'center_box'
            host.inspect_obj['proj_method'] = 'down_h_2'

        if not self._current_rect: 
            self.lbl_status.setText("Draw a box first.")
            return
        
        self._clear_markers()
        rect = self._current_rect
        cx = rect.x() + rect.width()/2
        cy = rect.y() + rect.height()/2
        
        self._ref_point_cctv = (cx, cy)
        
        # 1. Extract 4 corners of rect
        x1, y1 = rect.left(), rect.top()
        x2, y2 = rect.right(), rect.bottom()
        corners_cctv = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        trapezoid_sat = []
        for x, y in corners_cctv:
            if self._K is not None:
                src = np.array([[[x, y]]], dtype=np.float64)
                dst = cv2.undistortPoints(src, self._K, self._D, P=self._new_K)
                u, v = dst[0,0]
            else:
                u, v = x, y
            src_h = np.array([[[u, v]]], dtype=np.float64)
            pt_sat_corner = cv2.perspectiveTransform(src_h, self._H)[0,0]
            trapezoid_sat.append(pt_sat_corner)
            
        self._trapezoid_sat_corners = np.array(trapezoid_sat)
        
        # 2. Fit the initial box
        self._fit_init_box_to_trapezoid()
        self._fitted_h = DEFAULT_BOX_H
        self.spin_h.setValue(DEFAULT_BOX_H)
        
        if self._K is not None:
            src = np.array([[[cx, cy]]], dtype=np.float64)
            dst = cv2.undistortPoints(src, self._K, self._D, P=self._new_K)
            u_undist, v_undist = dst[0,0]
        else:
            u_undist, v_undist = cx, cy
            
        src_h = np.array([[[u_undist, v_undist]]], dtype=np.float64)
        pt_sat = cv2.perspectiveTransform(src_h, self._H)[0,0]
        
        self._proj_point_sat = pt_sat
        self._gc_point_cctv = self._sat_to_cctv(pt_sat)
        
        self._draw_marker(self.view_cctv, cx, cy, "Ref", Qt.red)
        dist = np.linalg.norm(np.array([cx, cy]) - np.array(self._gc_point_cctv))
        if dist > 2: self._draw_marker(self.view_cctv, self._gc_point_cctv[0], self._gc_point_cctv[1], "GC", Qt.green)
            
        self._draw_marker(self.view_sat, pt_sat[0], pt_sat[1], "PROJ", Qt.cyan)
        
        self._refresh_visuals()

    def _on_toggle_floor_box(self, checked):
        self._draw_floor_box()

    def _toggle_heading_mode(self, checked):
        self.slider_head.setEnabled(not checked)
        if not checked:
            self._heading_deg = self.slider_head.value()
            self._draw_floor_box()
            self.lbl_status.setText("Manual heading mode")
        else:
            if not getattr(self, '_svg_parser', None) or not getattr(self._svg_parser, 'valid', False):
                self.lbl_status.setText("Auto Heading enabled but no SVG loaded or no segments found.")
            elif not self._svg_parser.orientation_segments:
                self.lbl_status.setText("Auto Heading enabled but SVG has no guideline segments.")
            else:
                self.lbl_status.setText("Auto Heading enabled.")
            self._draw_floor_box()

    def _on_heading_changed(self, val):
        self._heading_deg = float(val)
        if self._current_rect is not None:
            self._fit_init_box_to_trapezoid()
        self._refresh_visuals()

    def _draw_floor_box(self):
        if self._proj_point_sat is None: return
        
        if self._highlight_line:
            try:
                if self._highlight_line.scene() == self.view_sat.scene(): self.view_sat.scene().removeItem(self._highlight_line)
            except RuntimeError: pass
            self._highlight_line = None

        if getattr(self, '_trapezoid_poly_item', None):
            try:
                if self._trapezoid_poly_item.scene() == self.view_sat.scene(): self.view_sat.scene().removeItem(self._trapezoid_poly_item)
            except RuntimeError: pass
            self._trapezoid_poly_item = None
            
        if hasattr(self, '_trapezoid_sat_corners'):
            poly = QGraphicsPolygonItem()
            qp = QPolygonF([QPointF(x, y) for x, y in self._trapezoid_sat_corners])
            poly.setPolygon(qp)
            poly.setPen(QPen(Qt.red, 2, Qt.DashLine))
            poly.setBrush(QBrush(QColor(255, 0, 0, 30)))
            poly.setZValue(14)
            self.view_sat.scene().addItem(poly)
            self._trapezoid_poly_item = poly

        if self.chk_auto_head.isChecked() and self._svg_parser:
            if hasattr(self, '_debug_items') and self._debug_items:
                for it in self._debug_items:
                    try:
                        if it.scene() == self.view_sat.scene(): self.view_sat.scene().removeItem(it)
                    except Exception:
                        pass
            self._debug_items = []
            pen_debug = QPen(QColor(200, 200, 200, 50), 1)
            for p1, p2 in self._svg_parser.orientation_segments:
                try:
                    l = self.view_sat.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen_debug)
                    l.setZValue(5)
                    self._debug_items.append(l)
                except Exception:
                    pass

        cx = getattr(self, '_fitted_cx', self._proj_point_sat[0])
        cy = getattr(self, '_fitted_cy', self._proj_point_sat[1])

        if self.chk_auto_head.isChecked() and self._svg_parser:
            heading, p1, p2 = self._svg_parser.get_nearest_heading_info(np.array([cx, cy]))
            if heading is not None:
                self._heading_deg = heading
                pen = QPen(Qt.magenta, 4)
                self._highlight_line = self.view_sat.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen)
                self._highlight_line.setZValue(100) 
            else:
                self.lbl_status.setText("No SVG guidelines found nearby.")
        
        w_m = getattr(self, '_fitted_w', self.spin_w.value())
        l_m = getattr(self, '_fitted_l', self.spin_l.value())
        w_px = w_m * self._px_per_m; l_px = l_m * self._px_per_m
        dx, dy = l_px/2.0, w_px/2.0
        corners = [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]
        
        rad = math.radians(self._heading_deg)
        c, s = math.cos(rad), math.sin(rad)
        rot_corners = []
        for x, y in corners:
            rx = x*c - y*s + cx; ry = x*s + y*c + cy
            rot_corners.append((rx, ry))
            
        if self._floor_poly:
            try:
                if self._floor_poly.scene() == self.view_sat.scene(): self.view_sat.scene().removeItem(self._floor_poly)
            except RuntimeError: pass
            self._floor_poly = None

        self._floor_corners_sat = rot_corners

        if getattr(self, 'chk_show_floor_box', None) and self.chk_show_floor_box.isChecked():
            poly = QGraphicsPolygonItem()
            qp = QPolygonF([QPointF(x, y) for x, y in rot_corners])
            poly.setPolygon(qp)
            poly.setPen(QPen(Qt.green, 2)); poly.setBrush(QBrush(QColor(0, 255, 0, 80)))
            poly.setZValue(15) 
            self.view_sat.scene().addItem(poly)
            self._floor_poly = poly

        if self._show_3d_active:
            self._on_show_3d()

    def _sat_to_cctv(self, pt_sat):
        src = np.array([[[pt_sat[0], pt_sat[1]]]], dtype=np.float64)
        pt_undist_px = cv2.perspectiveTransform(src, self._H_inv)[0,0]
        fx, fy = self._new_K[0,0], self._new_K[1,1]
        cx, cy = self._new_K[0,2], self._new_K[1,2]
        x_n = (pt_undist_px[0] - cx) / fx; y_n = (pt_undist_px[1] - cy) / fy
        obj_pts = np.array([[[x_n, y_n, 1.0]]], dtype=np.float32)
        img_pts, _ = cv2.projectPoints(obj_pts, (0,0,0), (0,0,0), self._K, self._D)
        return img_pts[0,0]

    def _on_show_3d(self):
        if not hasattr(self, '_floor_corners_sat'): return
        scene = self.view_cctv.scene()
        for item in self._wireframe_items:
            try:
                if item.scene() == scene: scene.removeItem(item)
            except RuntimeError: pass
        self._wireframe_items = []
        
        cctv_floor = [self._sat_to_cctv(pt) for pt in self._floor_corners_sat]
            
        def add_line(p1, p2, pen):
            l = scene.addLine(p1[0], p1[1], p2[0], p2[1], pen); l.setZValue(20)
            self._wireframe_items.append(l)

        for i in range(4):
            if i in [0, 2]: # Length edges
                pf = QPen(Qt.green, 2)
            elif i == 1: # Front width edge
                pf = QPen(QColor(255, 165, 0), 4)
            else: # Back width edge
                pf = QPen(Qt.cyan, 2)
            add_line(cctv_floor[i], cctv_floor[(i+1)%4], pf)

    def _clear_markers(self):
        s = self.view_sat.scene()
        if s:
            for i in self._sat_markers:
                try:
                    if i.scene() == s: s.removeItem(i)
                except RuntimeError: pass
        self._sat_markers = []
        c = self.view_cctv.scene()
        if c:
            for i in self._cctv_markers:
                try:
                    if i.scene() == c: c.removeItem(i)
                except RuntimeError: pass
        self._cctv_markers = []
        if self._floor_poly:
            try:
                if self._floor_poly.scene(): self._floor_poly.scene().removeItem(self._floor_poly)
            except RuntimeError: pass
            self._floor_poly = None
        if not self._show_3d_active:
            for w in self._wireframe_items:
                try:
                    if w.scene(): w.scene().removeItem(w)
                except RuntimeError: pass
            self._wireframe_items = []
        if self._highlight_line:
            try:
                if self._highlight_line.scene(): self._highlight_line.scene().removeItem(self._highlight_line)
            except RuntimeError: pass
            self._highlight_line = None
        if getattr(self, '_trapezoid_poly_item', None):
            try:
                if self._trapezoid_poly_item.scene(): self._trapezoid_poly_item.scene().removeItem(self._trapezoid_poly_item)
            except RuntimeError: pass
            self._trapezoid_poly_item = None

    def _on_toggle_3d(self, checked):
        self._show_3d_active = checked
        if checked:
            self.lbl_status.setText("3D box shown")
            try:
                self._on_show_3d()
            except Exception:
                pass
        else:
            self.lbl_status.setText("3D box hidden")
            for item in list(self._wireframe_items):
                try:
                    if item.scene(): item.scene().removeItem(item)
                except Exception:
                    pass
            self._wireframe_items = []

    def _on_toggle_roi(self, checked):
        if checked and self._roi_overlay: self.view_cctv.set_overlay(self._roi_overlay)
        else: self.view_cctv.clear_overlay()

    def _on_alpha_changed(self, val):
        if self._svg_item: self._svg_item.setOpacity(val / 100.0)

    def _on_save_config(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        
        obj = host.inspect_obj
        code = obj.get("meta", {}).get("location_code", "UNKNOWN")
        proj_root = getattr(self, "project_root", None) or os.getcwd()
        path = os.path.join(proj_root, "location", code, f"G_projection_{code}.json")
        try:
            from ..config import save_config
            save_config(path, obj)
            self.lbl_status.setText(f"Saved config to {path}")
        except Exception as e:
            self.lbl_status.setText(f"Save failed: {e}")

    def _draw_marker(self, viewer, x, y, label, color):
        scene = viewer.scene()
        el = scene.addEllipse(x-3, y-3, 6, 6, QPen(color, 2), QBrush(color))
        el.setZValue(25) 
        t = scene.addSimpleText(label)
        t.setBrush(QBrush(color))
        t.setPos(x+5, y-10)
        t.setZValue(25)
        
        visible = getattr(self, 'btn_toggle_pts', None) is None or self.btn_toggle_pts.isChecked()
        el.setVisible(visible)
        t.setVisible(visible)
        
        if viewer == self.view_sat: self._sat_markers.extend([el, t])
        else: self._cctv_markers.extend([el, t])

    def _on_toggle_points(self, checked):
        for item in self._sat_markers + self._cctv_markers:
            item.setVisible(checked)

    def _draw_line(self, viewer, p1, p2, color, width=2):
        scene = viewer.scene()
        l = scene.addLine(p1[0], p1[1], p2[0], p2[1], QPen(color, width))
        l.setZValue(15)
        if viewer == self.view_sat: self._sat_markers.append(l)
