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

## Help

Open **Help** inside the app for workflows, examples and solver configuration.

## Topology workflow

Larger voxel studies use a compiled matrix-free elasticity solver with
multigrid preconditioning. CAD charts use compiled inverse sampling and bounded
parallel preparation. TopOpt FEA selects MKL PARDISO for supported linear
tetrahedral models; **FEA Solver** also offers CalculiX explicitly.

Lattice sizing reuses fixed geometry and, where a cached periodic material law
is available, starts with a coarse continuum proposal. It then checks the actual
lattice material with solid FEA. Final acceptance still requires geometry,
equilibrium, stress and mesh-convergence checks; a generated lattice alone does
not establish its load capacity.

CAD finishing fits source points and a filtered physical normal field together
with neighboring surface normals. Displacement follows source voxel spacing and
source-fit accuracy. Bounded shared-curve/UV repair and physical-curvature
fairing are available, with separate checks for seam errors and interior
ripples. A proposal replaces the current solid only after native geometry,
visible-surface and STEP checks pass. The
[reconstruction notes](pylcss/cae/design_studio/topology_optimization/geometry/RECONSTRUCTION.md)
describe the method, spatial diagnostics and T-spline alternatives. FEA stores
hotspot coordinates and can continue a matching saved analysis; changed geometry
or loads invalidate that response.

## License

Source available under the [PolyForm Shield License 1.0.0](LICENSE).
Third-party licenses are listed in [NOTICE](NOTICE).
