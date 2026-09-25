"""
main.py – Entry point.

Run with:
    python main.py

Optional environment variable:
    IPM_PROJECT_ROOT – path to the project root that contains the
                       'location/' directory.  Defaults to the directory
                       containing this file.
"""
import os
import sys


def main():
    project_root = os.environ.get("IPM_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("IPM Tool")

    # Apply dark theme if qdarktheme is available
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    except ImportError:
        # Fall back to a minimal built-in dark palette
        from PyQt5.QtGui import QPalette, QColor
        palette = QPalette()
        palette.setColor(QPalette.Window,          QColor(45,  45,  45))
        palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
        palette.setColor(QPalette.Base,            QColor(30,  30,  30))
        palette.setColor(QPalette.AlternateBase,   QColor(53,  53,  53))
        palette.setColor(QPalette.ToolTipBase,     QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText,     QColor(0,   0,   0))
        palette.setColor(QPalette.Text,            QColor(220, 220, 220))
        palette.setColor(QPalette.Button,          QColor(53,  53,  53))
        palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText,      QColor(255, 0,   0))
        palette.setColor(QPalette.Link,            QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight,       QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0,   0,   0))
        app.setPalette(palette)

    from src.app import MainWindow
    window = MainWindow(project_root=project_root)
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
