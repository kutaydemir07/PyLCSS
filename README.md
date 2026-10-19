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

Download the Windows installer [PyLCSS-3.0.0-Setup-x64.exe](PyLCSS-3.0.0-Setup-x64.exe) directly from the repository.
Choose **CAE**, **MDAO**, or both during setup, then launch **PyLCSS** from the
Start Menu. External solvers and FreeCAD are optional setup components.

### From source

Requires Python 3.11+; Windows 10/11 is the primary platform.

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

For only the crash solver, run `python scripts/install_solvers.py --only opencourant`.

CAD FEA uses the standalone Gmsh mesher when available. The Windows installer
includes it; for a source checkout, run `python scripts/install_solvers.py --only gmsh`.
Set `PYLCSS_GMSH_EXE` to use another installed Gmsh executable. CAD meshing and
analysis run in child processes and retain the fitted CAD and exact interfaces.

FEA and crash solvers use at most four CPU threads by default. To tune a model,
set `PYLCSS_CALCULIX_THREADS` (FEA) or `PYLCSS_OPENCOURANT_THREADS` (crash) to a
positive integer. Explicit limits are capped to the available logical CPUs and
apply only to solver processes.

## Help

Open **Help** inside the app for workflows, examples and solver configuration.

## License

Source available under the [PolyForm Shield License 1.0.0](LICENSE).
Third-party licenses are listed in [NOTICE](NOTICE).
