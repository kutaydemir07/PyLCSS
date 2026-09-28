"""Choose an installed PyLCSS workspace before loading engineering libraries."""
import logging
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from pylcss.workspaces import create_window

class StartPage(QtWidgets.QMainWindow):
    def __init__(self, available):
        super().__init__()
        QtWidgets.QApplication.instance().setQuitOnLastWindowClosed(False)
        self.setWindowTitle("PyLCSS — Choose a workspace")
        self.setWindowIcon(QtGui.QIcon(str(Path(__file__).with_name("icon.png"))))
        self.resize(900, 600)
        self._windows = {}
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(80, 60, 80, 60)
        layout.setSpacing(24)
        layout.addStretch()
        title = QtWidgets.QLabel("Welcome to PyLCSS")
        title.setStyleSheet("font-size: 32px; font-weight: 600;")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel("Choose the workspace you want to open."))
        for name, description in (
            ("mdao", "System modeling, optimization, sensitivity and surrogate models"),
            ("cae", "CAD, meshing, simulation and topology optimization"),
        ):
            button = QtWidgets.QPushButton(f"PyLCSS {name.upper()}\n{description}")
            button.setMinimumHeight(90)
            button.setEnabled(name in available)
            if name not in available:
                button.setText(button.text() + "\nNot installed — run setup to add this workspace")
            button.clicked.connect(lambda checked=False, selected=name: self.open_workspace(selected))
            layout.addWidget(button)
        layout.addStretch()

    def open_workspace(self, name):
        if name not in self._windows:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                window = create_window(name)
                window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
                window.destroyed.connect(lambda: self.workspace_closed(name))
                self._windows[name] = window
            except Exception as exc:
                logging.getLogger(__name__).exception("Workspace startup failed")
                QtWidgets.QMessageBox.critical(self, "Could not open workspace", str(exc))
                return
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        self._windows[name].showMaximized()
        self._windows[name].raise_()
        self.hide()

    def workspace_closed(self, name):
        self._windows.pop(name, None)
        self.show()

    def closeEvent(self, event):
        if any(not window.close() for window in list(self._windows.values())):
            event.ignore()
            return
        QtWidgets.QApplication.instance().quit()
        event.accept()
