"""
app.py – Main application window.

QMainWindow that hosts the CalibrationTab.  All the original
qdarktheme / overlay logic is preserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget,
)

from .calibration.tab import CalibrationTab


class MainWindow(QMainWindow):
    """Top-level window with a tab bar (currently one tab: Calibration)."""

    def __init__(self, project_root: str):
        super().__init__()
        self.setWindowTitle("Infer-3Dbbox-From-2Dbbox")
        self.resize(1400, 900)

        self._tabs = QTabWidget()
        self._calib_tab = CalibrationTab(project_root=project_root)
        self._tabs.addTab(self._calib_tab, "Fit 3D box to 2D box")

        self.setCentralWidget(self._tabs)

        # Maximised overlay label (hidden by default)
        self._overlay = QLabel(self)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setStyleSheet(
            "background: rgba(0,0,0,0.55); color: white; font-size: 18px;"
        )
        self._overlay.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())
