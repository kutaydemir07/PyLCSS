<!--
Copyright (c) 2026 Kutay Demir.
Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
-->

# PyLCSS 3.0.0

**Low-Code System Solutions** — a desktop workbench for engineering design,
simulation and optimization through visual, node-based workflows.

- **CAE:** parametric CAD, structural and modal FEA, crash analysis, topology
  optimization and lattice design.
- **MDAO:** system modeling, solution spaces, sensitivity analysis, surrogate
  modeling and multi-objective optimization.

## Installation

### Windows

Download the Windows installer from [Releases](https://github.com/kutaydemir07/PyLCSS/releases).
Choose **CAE**, **MDAO**, or both during setup, then launch **PyLCSS** from the
Start Menu. External solvers and FreeCAD are optional setup components.

### From source

Requires Python 3.10+; Windows 10/11 is the primary platform.

```powershell
git clone https://github.com/kutaydemir07/PyLCSS.git
cd PyLCSS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[mdao,cae]"
python -m pylcss.main
```

For one workspace, install `.[mdao]` or `.[cae]` instead. To download CalculiX,
OpenCourant and FreeCAD, run `python scripts/install_solvers.py`.

Install only the crash solver with `python scripts/install_solvers.py --only
opencourant`. The installer verifies the pinned upstream archive and configures
Starter, Engine, animation conversion, and time-history conversion automatically.
Run `python -m pytest tests/test_opencourant_integration.py` to check the integration
when the CAE dependencies and solver are installed.

## Help

Open **Help** inside the app for workflows, examples and solver configuration.

## License

Source available under the [PolyForm Shield License 1.0.0](LICENSE).
Third-party licenses are listed in [NOTICE](NOTICE).
