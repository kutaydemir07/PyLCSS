"""Standalone CAE workspace for CAD and simulation."""
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from pylcss.cae.user_interface.cad import ProfessionalCadApp
from pylcss.user_interface.common import apply_theme, current_theme


class CAEMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyLCSS CAE")
        self.setWindowIcon(QtGui.QIcon(str(Path(__file__).parents[2] / "user_interface/icon.png")))
        self.resize(1600, 900)
        self.cad_widget = ProfessionalCadApp()
        self.setCentralWidget(self.cad_widget)
        theme_menu = self.menuBar().addMenu("Theme")
        for name in ("dark", "light"):
            action = theme_menu.addAction(name.title())
            action.triggered.connect(lambda checked=False, selected=name: self._set_theme(selected))
        self._help_dialog = None
        help_action = self.menuBar().addMenu("Help").addAction("User Guide")
        help_action.setShortcut("F1")
        help_action.triggered.connect(self._show_help)

    def _show_help(self):
        if self._help_dialog is None:
            from pylcss.user_interface.help import HelpWidget
            self._help_dialog = QtWidgets.QDialog(self)
            self._help_dialog.setWindowTitle("PyLCSS User Guide")
            self._help_dialog.resize(1050, 750)
            layout = QtWidgets.QVBoxLayout(self._help_dialog)
            self._help_widget = HelpWidget()
            layout.addWidget(self._help_widget)
        self._help_widget.apply_theme(current_theme())
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    def _set_theme(self, name):
        apply_theme(name)
        self.cad_widget.apply_theme(name)
        if self._help_dialog is not None:
            self._help_widget.apply_theme(name)

    def save_project(self):
        self.cad_widget._save_project()

    def load_project(self):
        self.cad_widget._open_project()

    def closeEvent(self, event):
        if not self.cad_widget._ensure_idle_for_io("closing PyLCSS CAE"):
            event.ignore()
            return
        super().closeEvent(event)
