"""Workspace discovery and lazy application loading."""
import json
import os
from pathlib import Path

WORKSPACES = ("mdao", "cae")

def installed_workspaces():
    root = os.environ.get("PYLCSS_INSTALL_ROOT")
    if root:
        receipt = Path(root) / "install" / "installation.json"
        if receipt.is_file():
            document = json.loads(receipt.read_text(encoding="utf-8-sig"))
            return tuple(name for name in WORKSPACES
                         if name in document.get("workspaces", WORKSPACES))
    # Source installations record their selected extras via installed dependencies.
    from importlib.util import find_spec
    requirements = {"mdao": ("nevergrad", "torch", "SALib"),
                    "cae": ("cadquery", "vtk", "pymoto")}
    return tuple(name for name, modules in requirements.items()
                 if all(find_spec(module) is not None for module in modules))

def create_window(workspace):
    if workspace == "mdao":
        from pylcss.mdao.user_interface.main_application_window import MainWindow
        return MainWindow()
    if workspace == "cae":
        from pylcss.cae.user_interface.main_window import CAEMainWindow
        return CAEMainWindow()
    raise ValueError(f"Unknown workspace: {workspace}")
