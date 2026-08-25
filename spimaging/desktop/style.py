"""Calm, high-contrast visual system for the Windows workbench."""

APP_STYLESHEET = r"""
QWidget {
    color: #1b2430;
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
}
QMainWindow, QWidget#appRoot { background: #f4f6f8; }
QFrame#sidebar { background: #142334; border: none; }
QLabel#brandTitle { color: white; font-size: 21px; font-weight: 700; }
QLabel#brandSubtitle { color: #9db0c5; font-size: 11px; }
QPushButton#navButton {
    background: transparent; color: #c9d6e3; border: none; border-radius: 7px;
    padding: 10px 14px; text-align: left; font-weight: 600;
}
QPushButton#navButton:hover { background: #20364d; color: white; }
QPushButton#navButton:checked { background: #2667a8; color: white; }
QFrame#topbar { background: white; border-bottom: 1px solid #dce2e8; }
QLabel#pageTitle { font-size: 24px; font-weight: 700; color: #15263a; }
QLabel#pageSubtitle { color: #66788a; }
QFrame#card {
    background: white; border: 1px solid #dfe5eb; border-radius: 10px;
}
QLabel#cardTitle { font-size: 16px; font-weight: 700; color: #1a2d42; }
QLabel#muted { color: #697b8d; }
QPushButton {
    background: #ffffff; border: 1px solid #c9d3dc; border-radius: 6px;
    padding: 7px 13px;
}
QPushButton:hover { border-color: #2878bd; color: #175f9e; }
QPushButton:disabled { color: #9da7af; background: #f1f3f5; border-color: #e0e4e7; }
QPushButton#primaryButton { background: #176fb0; border-color: #176fb0; color: white; font-weight: 700; }
QPushButton#primaryButton:hover { background: #0f5f9b; }
QPushButton#dangerButton { color: #a72929; border-color: #d9a5a5; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget, QTreeWidget {
    background: white; border: 1px solid #cbd5de; border-radius: 5px; padding: 5px;
    selection-background-color: #2978b9;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #2878bd; }
QHeaderView::section { background: #edf2f6; border: none; border-bottom: 1px solid #d4dce3; padding: 7px; font-weight: 600; }
QProgressBar { background: #e5eaf0; border: none; border-radius: 5px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #2682bd; border-radius: 5px; }
QGroupBox { border: 1px solid #dce3e9; border-radius: 7px; margin-top: 10px; padding-top: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QScrollArea { border: none; background: transparent; }
QLabel#statusBadge { border-radius: 9px; padding: 3px 9px; font-weight: 700; background: #e7edf3; color: #42566b; }
QLabel#statusBadge[status="running"] { background: #dceffd; color: #075f9d; }
QLabel#statusBadge[status="succeeded"] { background: #dcf5e8; color: #17663d; }
QLabel#statusBadge[status="failed"] { background: #fde2e2; color: #9b2727; }
QLabel#statusBadge[status="cancelled"], QLabel#statusBadge[status="interrupted"] { background: #f6ead4; color: #805716; }
QFrame#imagePanel { background: #f8fafb; border: 1px solid #dce3e8; border-radius: 7px; }
QToolTip { background: #142334; color: white; border: none; padding: 5px; }
"""


__all__ = ["APP_STYLESHEET"]
