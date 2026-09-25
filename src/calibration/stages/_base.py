"""
_base.py – Shared viewer widgets and OpenCV utilities.

Every calibration stage imports its image-viewer classes from here
instead of duplicating them or importing across stage files.
"""
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)


# ---------------------------------------------------------------------------
# Core pan/zoom viewer
# ---------------------------------------------------------------------------

class ImageViewer(QGraphicsView):
    """Pan-and-zoom image viewer backed by QGraphicsView.

    Emits ``clicked(float, float)`` with scene-space coordinates when the
    user left-clicks on an image.
    """

    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = None
        self._overlay_item = None
        self._zoom = 0
        self.setRenderHints(self.renderHints() | Qt.SmoothTransformation)
        self.setInteractive(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._needs_fit = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_pixmap(self, pixmap: QPixmap):
        """Replace the scene content with *pixmap*."""
        self._needs_fit = True
        _scene = self.scene()
        # Remove tracked items before clear() so Qt doesn't delete the C++
        # objects while Python wrappers still hold dangling pointers.
        for attr in ("_pixmap_item", "_overlay_item"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    if item.scene() == _scene:
                        _scene.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)
        _scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        _scene.addItem(self._pixmap_item)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self._zoom = 0

    def change_background_pixmap(self, pixmap: QPixmap):
        """Update only the background pixmap without clearing the scene or deleting overlays."""
        if self._pixmap_item is not None:
            self._pixmap_item.setPixmap(pixmap)
            self.setSceneRect(self._pixmap_item.boundingRect())
        else:
            self.load_pixmap(pixmap)
            
    def fitToView(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

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

    def set_overlay_rect(self, x: int, y: int, w: int, h: int):
        """Draw a green bounding-box overlay (used for ROI crop preview)."""
        try:
            if self._overlay_item is not None:
                try:
                    self.scene().removeItem(self._overlay_item)
                except Exception:
                    pass
                self._overlay_item = None
            rect = QGraphicsRectItem(x, y, w, h)
            pen = rect.pen()
            pen.setWidth(2)
            pen.setColor(QColor(0, 255, 0))
            rect.setPen(pen)
            rect.setBrush(QColor(0, 0, 0, 0))
            rect.setZValue(2)
            self._overlay_item = self.scene().addItem(rect)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, '_needs_fit', False):
            self.fitToView()
            self._needs_fit = False

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        try:
            if self._pixmap_item is not None and event.button() == Qt.LeftButton:
                pt = self.mapToScene(event.pos())
                self.clicked.emit(float(pt.x()), float(pt.y()))
        except Exception:
            pass
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Right-click variant (used for point placement in several stages)
# ---------------------------------------------------------------------------

class RightClickImageViewer(ImageViewer):
    """Variant of ImageViewer where *right*-click emits the ``clicked`` signal
    so left-click is always free for panning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._pixmap_item is not None and event.button() == Qt.RightButton:
            pt = self.mapToScene(event.pos())
            self.clicked.emit(float(pt.x()), float(pt.y()))
            event.accept()
        else:
            # Go straight to QGraphicsView (skip ImageViewer's left-click emit)
            QGraphicsView.mousePressEvent(self, event)


# ---------------------------------------------------------------------------
# OpenCV utilities shared across stages
# ---------------------------------------------------------------------------

def qimage_to_cv(qimg: QImage):
    """Convert a QImage (any format) to a BGR numpy array."""
    if qimg is None:
        return None
    img = qimg.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
    return arr[:, :, ::-1].copy()   # RGB → BGR


def cv_to_qimage(cv_bgr) -> QImage:
    """Convert a BGR numpy array to a QImage."""
    if cv_bgr is None:
        return None
    rgb = cv_bgr[:, :, ::-1]
    h, w, ch = rgb.shape
    return QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888).copy()


def remap_with_supersample(src, K, D, newcameramtx):
    """Undistort *src* using remap with optional supersampling for high-distortion lenses."""
    if src is None or not HAS_CV2:
        return src
    h, w = src.shape[:2]
    try:
        max_abs = float(np.max(np.abs(D))) if D is not None else 0.0
    except Exception:
        max_abs = 0.0

    scale = 3 if max_abs > 1.0 else (2 if max_abs > 0.6 else 1)

    try:
        if scale == 1:
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (w, h), cv2.CV_32FC1)
            return cv2.remap(src, mapx, mapy, interpolation=cv2.INTER_LANCZOS4)

        w2, h2 = int(w * scale), int(h * scale)
        src_up = cv2.resize(src, (w2, h2), interpolation=cv2.INTER_CUBIC)
        Ks = K.astype(np.float64).copy()
        Ns = newcameramtx.astype(np.float64).copy()
        for m in (Ks, Ns):
            m[0, 0] *= scale; m[1, 1] *= scale
            m[0, 2] *= scale; m[1, 2] *= scale
        mapx, mapy = cv2.initUndistortRectifyMap(Ks, D, None, Ns, (w2, h2), cv2.CV_32FC1)
        und_up = cv2.remap(src_up, mapx, mapy, interpolation=cv2.INTER_LANCZOS4)
        return cv2.resize(und_up, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        try:
            return cv2.undistort(src, K, D, None, newcameramtx)
        except Exception:
            return None
