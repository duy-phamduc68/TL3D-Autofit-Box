"""
tab.py – CalibrationTab

Central wizard coordinator.  All stage-switching logic lives here.
Key design rules (bug-fixes over the original tab_calibration.py):

  1. Stages are created ONCE and kept alive in self._stages[].
     We never call setParent(None) on them – that broke showEvent-based
     parent lookups and leaked C++ objects.

  2. host_tab is injected once at construction so stages never need to
     climb the parent chain themselves.

  3. Navigation is done exclusively via _go_to_stage(idx).  Stage
     proceed-buttons call host._go_to_stage() directly.

  4. The progress-bar timeline is updated from _go_to_stage so it is
     always in sync with the displayed stage.

  5. inspect_obj is a plain Python dict stored on the tab.  Stages read
     and write it via `host.inspect_obj`.
"""
import json
import os
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from .config import save_config, to_pretty_json, load_config, default_config
from .stages.final    import FinalStage

# ---------------------------------------------------------------------------
# Stage index constants (edit here if you add/remove stages)
# ---------------------------------------------------------------------------

STAGE_LABELS = [
    "Final Validation",       # 0
]




# ---------------------------------------------------------------------------
# Progress / timeline bar
# ---------------------------------------------------------------------------

class _ProgressBar(QWidget):
    """Compact horizontal stage progress indicator."""

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self._labels  = labels
        self._current = 0
        self._buttons: list[QPushButton] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        for i, lbl in enumerate(labels):
            btn = QPushButton(lbl)
            btn.setFixedHeight(28)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setProperty("stage_idx", i)
            btn.clicked.connect(self._on_clicked)
            self._buttons.append(btn)
            layout.addWidget(btn)

        self._refresh()

    def set_current(self, idx: int):
        self._current = max(0, min(idx, len(self._labels) - 1))
        self._refresh()

    def _refresh(self):
        for i, btn in enumerate(self._buttons):
            if i == self._current:
                btn.setStyleSheet(
                    "background-color: #2a84ff; color: white; font-weight: bold; border-radius: 3px;"
                )
            elif i < self._current:
                btn.setStyleSheet(
                    "background-color: #1a5599; color: #ccc; border-radius: 3px;"
                )
            else:
                btn.setStyleSheet(
                    "background-color: #3a3a3a; color: #888; border-radius: 3px;"
                )

    # Allow clicking earlier stages to jump back
    def _on_clicked(self):
        idx = self.sender().property("stage_idx")
        # Find the CalibrationTab ancestor
        p = self.parent()
        while p is not None:
            if isinstance(p, CalibrationTab):
                p._go_to_stage(idx)
                return
            p = p.parent()


# ---------------------------------------------------------------------------
# Main CalibrationTab
# ---------------------------------------------------------------------------

class CalibrationTab(QWidget):
    """Tab widget that hosts all calibration stages as a wizard."""

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root or os.getcwd()
        pr = self.project_root
        
        gpath = os.path.join(pr, "location", "SHINJUKU1", "G_projection_SHINJUKU1.json")
        if os.path.isfile(gpath):
            self.inspect_obj = load_config(gpath)
        else:
            self.inspect_obj = default_config("SHINJUKU1")

        self.current_step_index = 0

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---- Progress bar ----
        self.progress_bar = _ProgressBar(STAGE_LABELS, self)
        root_layout.addWidget(self.progress_bar)

        # ---- Stage stack ----
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, 1)

        # ---- Build all stages (created once, never destroyed) ----
        self._stages: list[QWidget] = [
            FinalStage(pr),         # 0
        ]

        # Add to stack and inject host reference
        for stage in self._stages:
            stage.host_tab = self           # type: ignore[attr-defined]
            self.stack.addWidget(stage)

        self._go_to_stage(0)

    # ------------------------------------------------------------------
    # Public API used by all stages
    # ------------------------------------------------------------------

    def _go_to_stage(self, idx: int):
        """Switch to *idx*, update progress bar, trigger showEvent."""
        if idx < 0 or idx >= len(self._stages):
            return
        self.current_step_index = idx
        self.stack.setCurrentIndex(idx)
        self.progress_bar.set_current(idx)

    # Legacy names kept for any old code that still calls them
    def _show_stage(self, idx: int):
        self._go_to_stage(idx)

    def _update_progress_to_index(self, idx: int):
        self.progress_bar.set_current(idx)
