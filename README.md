# PyLCSS: Low-Code System Solutions

<div align="center">

<img src="pylcss/user_interface/icon.png" width="48" alt="PyLCSS Logo">

<br/>

<a href="https://youtu.be/fQuLZ5LnxQs" target="_blank">
  <img src="https://img.youtube.com/vi/fQuLZ5LnxQs/maxresdefault.jpg" width="100%" alt="PyLCSS Video Review">
</a>

<img src="pylcss/user_interface/1778627783968.gif" width="100%" alt="PyLCSS Crash Demo">

<img src="pylcss/user_interface/topopt.png" width="48%" alt="Topology Optimization Simulation">
<img src="pylcss/user_interface/fea.png" width="48%" alt="Finite Element Analysis Solver">

<img src="pylcss/user_interface/cad_boeing747.png" width="48%" alt="Boeing 747 Parametric CAD">
<img src="pylcss/user_interface/cad_helicalgear.png" width="48%" alt="Helical Gear CAD Model">

**(Click the video thumbnail above to watch the demonstration on YouTube!)**

**Source-Available Engineering Simulation & Optimization Platform**

*Visual Modeling · Parametric CAD · Topology Optimization · FEA · Solution Spaces · Sensitivity Analysis · Surrogate AI · Multi-Objective Optimization*

[![License](https://img.shields.io/badge/License-PolyForm_Shield_1.0.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-2.2.0-orange.svg)]()
[![Slides](https://img.shields.io/badge/Slides-PDF-red.svg)](pylcss/user_interface/1773249914176.pdf)

</div>

---

## Overview

**PyLCSS** (Python Low-Code System Solutions) is an integrated product development environment for engineering design, enabling engineers to model and analyze multidisciplinary systems through an intuitive node-based visual interface, all within a single desktop application.

The core concept is the **Solution Space** approach for robust design: instead of seeking a single optimal point, it identifies box-shaped regions of valid designs that allow decoupled subsystem development, as introduced by:

> *Markus Zimmermann, Johannes Edler von Hoessle*, "Computing solution spaces for robust design", *Int. J. Numer. Meth. Engng.*, 2013. [DOI: 10.1002/nme.4450](https://doi.org/10.1002/nme.4450)

### Features

- **Parametric CAD** — Define geometry in Python (CadQuery) or draw it interactively in FreeCAD via a live bridge
- **FEA** — Static structural analysis via CalculiX with displacement and von Mises stress results visualised in the built-in VTK viewer
- **Topology Optimization** — SIMP-based voxel topology optimization via pyMOTO; direct STL/OBJ export of optimized geometry
- **Crash / Impact Simulation** — OpenRadioss explicit solver integration with animated VTK result playback
- **Solution Space Exploration** — Find all designs that satisfy your requirements, not just a single optimum; includes product family analysis to identify a common platform across variants
- **Multi-Objective Optimization** — 7 solvers: SLSQP, COBYLA, trust-constr, Differential Evolution, Nevergrad, NSGA-II, and Multi-Start
- **Global Sensitivity Analysis** — 4 methods: Sobol, Morris, FAST, and Delta (DMIM)
- **Surrogate Modelling** — 4 algorithms (MLP, Random Forest, Gradient Boosting, Gaussian Process / Kriging) with cross-validation, hyperparameter search, and feature importance
- **System Modelling** — Graph-based functional architecture editor for mapping requirements to subsystems
- **AI Assistant** — text-driven PydanticAI agent with 25 tools and multi-provider LLM support (OpenAI, Anthropic, Google, and OpenAI-compatible local servers like LM Studio / Ollama / vLLM)
- **Black-Box Integration** — Wrap any external solver (ANSYS, MATLAB, LS-DYNA, HPC scripts) in a simple `evaluate(x)` function

Detailed documentation on node types, workflows, and solver configuration is available in the **Help** widget inside the application.

---

## Installation

**Requirements:** Python 3.10+ · Windows 10/11 (macOS/Linux: experimental)

```bash
git clone <repository-url>
cd pylcss

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# Optional: download CalculiX, OpenRadioss, FreeCAD (interactive)
python scripts/install_solvers.py

python scripts/main.py
```

Or on Windows: double-click `run_gui.bat`.

External solvers (CalculiX, OpenRadioss, FreeCAD) are optional. PyLCSS opens cleanly without them; solver nodes remain available for deck-only preparation and show their detected runtime status in the component-library tooltip. Attempting a full solve without the required executable reports a node error. The external tools are governed by their own upstream licenses (CalculiX: GPL, OpenRadioss: AGPL-3.0, FreeCAD: LGPL-2.1+).

---

## License

Licensed under the **PolyForm Shield License 1.0.0**.

**Allowed:** Personal use, academic research, internal business use.
**Restricted:** You cannot use this software to build a competing product or service.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full details.

<div align="center">
<sub>Copyright © 2026 Kutay Demir. All rights reserved.</sub>
</div>
