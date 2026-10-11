<!--
Copyright (c) 2026 Kutay Demir.
Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
-->

# PyLCSS: Low-Code System Solutions

<div align="center">

<img src="pylcss/user_interface/icon.png" width="48" alt="PyLCSS Logo">

**Source-Available Engineering Simulation & Optimization Platform**

*Visual Modeling · Parametric CAD · Topology &amp; Lattice Optimization · FEA · Solution Spaces · Sensitivity Analysis · Surrogate AI · Multi-Objective Optimization*

[![License](https://img.shields.io/badge/License-PolyForm_Shield_1.0.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-2.2.0.2-orange.svg)]()
[![Slides](https://img.shields.io/badge/Slides-PDF-red.svg)](pylcss/user_interface/1773249914176.pdf)

</div>

---

## Overview

**PyLCSS** (Python Low-Code System Solutions) is an integrated product development environment for engineering design, enabling engineers to model and analyze multidisciplinary systems through an intuitive node-based visual interface, all within a single desktop application.

The core concept is the **Solution Space** approach for robust design: instead of seeking a single optimal point, it identifies box-shaped regions of valid designs that allow decoupled subsystem development, as introduced by:

> *Markus Zimmermann, Johannes Edler von Hoessle*, "Computing solution spaces for robust design", *Int. J. Numer. Meth. Engng.*, 2013. [DOI: 10.1002/nme.4450](https://doi.org/10.1002/nme.4450)

### Features

- **Parametric CAD** — CadQuery Python scripts and linked FreeCAD parametric parts
- **FEA & Modal Analysis** — CalculiX static stress and natural frequency extraction with 3D mode shape eigenvector visualization
- **Loads & Boundary Conditions** — Force, face pressure, gravity, bearing contact loads (cosine compressive traction), and multi-case weighting
- **Topology Optimization** — Structural, thermal, and multibody SIMP workflows with automated AMRTO B-rep reconstruction (watertight STEP/IGES)
- **Lattice Optimization** — Variable-density Gyroid, Schwarz Primitive, BCC, Octet, and Honeycomb cells with STEP, STL, and 3MF beam lattice output
- **Impact & Crashworthiness** — OpenRadioss transient simulation, crashworthiness KPI dashboard (CFE, SEA, force, stroke), physical energy gates, and interactive time-history curves
- **Solution Spaces** — Decoupled feasible intervals and product-family exploration with 1-D, 2-D, and 3-D interactive visualization
- **Multi-Objective Optimization** — SLSQP, Nevergrad, Multi-Start, and NSGA-II Pareto optimization with objective normalization
- **Sensitivity Analysis** — SOTA Sobol' global variance decomposition (Saltelli sequence sampling), $S_1$, $S_T$, and pairwise $S_2$ interaction heatmaps
- **Surrogate AI & Auto Mode** — Scientific Auto Mode (Kriging ARD, KPLS, RBF splines, Sparse PCE, SVR, Goel-Haftka ensemble), custom deep learning architectures (ResNet, Fourier embeddings), and tabular data import
- **System Modeling** — Graph-based functional architecture and linked Design Studio CAD study evaluation
- **Black-Box Integration** — Wrap external tools and custom Python routines into reusable nodes

Detailed documentation on node types, workflows, and solver configuration is available in the **Help** widget inside the application.

---

## Installation

### Windows installer

Run `PyLCSS-2.2.0.2-Setup-x64.exe`. Setup displays the PyLCSS license and a
separate third-party notice, then installs an isolated Python runtime, the
selected Python packages, and Start Menu/desktop shortcuts. Choose **MDAO**,
**CAE**, or **both** on the component selection page. CalculiX,
OpenRadioss, FreeCAD, and AMRTO are downloaded only when their optional tasks are
selected. Normal startup is the terminal-free `PyLCSS.exe` at the application
root. Installed copies check the official GitHub Releases feed at most once a
day. When a newer stable version is available, the launcher offers to download
the matching Windows setup and SHA-256 file, requires a trusted Authenticode
signature, and hands the upgrade to Setup. Setup replaces release-managed
application files and installs any new or changed locked dependencies. The
check can be suppressed with `PyLCSS.exe --skip-update-check` or the
`PYLCSS_DISABLE_UPDATE_CHECK=1` environment variable; use
`PyLCSS.exe --check-for-updates` to force a check. Developer checkouts never
auto-update. Normal application startup never runs pip or modifies the system
PATH.

The reproducible setup recipe and signing-ready installer source are under
`installer/windows/`. Windows CPython 3.12 package artifacts are pinned by
version and SHA-256 in `requirements-windows-py312.lock`.

For an update to be discoverable, publish a stable GitHub Release tagged
`vX.Y.Z` (or `X.Y.Z`) and attach both
`PyLCSS-X.Y.Z-Setup-x64.exe` and its generated `.exe.sha256` file. Public
release artifacts must be built with `build_windows_installer.ps1
-RequireSignature` and a trusted code-signing certificate; unsigned online
updates are rejected.

### Developer installation

**Requirements:** Python 3.10+ · Windows 10/11 (macOS/Linux: experimental)

```bash
git clone <repository-url>
cd pylcss

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Choose one workspace or both:
pip install -e ".[mdao]"
# pip install -e ".[cae]"
# pip install -e ".[mdao,cae]"

# Optional: download external tools (CalculiX, OpenRadioss, FreeCAD, AMRTO)
python scripts/install_solvers.py

python -m pylcss.main
```

Startup shows a workspace chooser. You can also launch directly with
`python -m pylcss.mdao`, `python -m pylcss.cae`, or
`PyLCSS.exe --workspace mdao` / `--workspace cae`.

MDAO code lives under `pylcss/mdao/` (system modeling, optimization,
sensitivity, solution spaces, and surrogate modeling). CAE code lives under
`pylcss/cae/` (Design Studio, meshing, simulation, topology optimization,
and solver backends). Shared configuration, I/O, themes, and the start page
remain under `pylcss/`. Legacy Python import paths still work for saved models.
CAD-backed MDAO models require both workspaces. The Windows installer records
the selected workspaces; rerun setup to add one. Source installations can use
`requirements-mdao.txt`, `requirements-cae.txt`, or `requirements.txt` for both.
The installer uses the corresponding hashed Windows dependency lock. Shared
source files are shipped together, but dependencies and launch choices follow
the selected components. Removing a workspace selection during an upgrade
hides it; previously downloaded packages are retained in the shared runtime.

External solvers and tools (CalculiX, OpenRadioss, FreeCAD, AMRTO) are optional. PyLCSS opens cleanly without them; solver nodes remain available for deck-only preparation and show their detected runtime status in the component-library tooltip. Attempting a full solve without the required executable reports a node error. The external tools are governed by their own upstream licenses (CalculiX: GPL, OpenRadioss: AGPL-3.0, FreeCAD: LGPL-2.1+, AMRTO: Research License / BSD / MIT).

---

## License

Licensed under the **PolyForm Shield License 1.0.0**.

Examples of permitted noncompeting uses include personal projects, academic
research, and internal business work. Providing a product or service that
competes with PyLCSS is not permitted, even if that competing offering is free.
The complete terms and definitions in `LICENSE` control; these examples are not
an independent license grant.

See [LICENSE](LICENSE) for the complete standardized PyLCSS terms and
[NOTICE](NOTICE) for separately licensed dependencies and optional components.
See [PRIVACY.md](PRIVACY.md) for data handling and [SECURITY.md](SECURITY.md)
before opening projects or serialized models from other people.

<div align="center">
<sub>Copyright © 2026 Kutay Demir. All rights reserved.</sub>
</div>
