# PyLCSS: Low-Code System Solutions

<div align="center">

![PyLCSS Banner](pylcss/user_interface/icon.png)

**🚀 Source-Available Engineering Simulation & Optimization Platform**

*Visual Modeling • CAD and FEM • Solution Space Exploration • AI-Powered Surrogates • Multi-Objective Optimization*

[![License](https://img.shields.io/badge/License-PolyForm_Shield_1.0.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production_Ready-orange.svg)]()

</div>

---

## 🎯 Overview

**PyLCSS** (Python Low-Code System Solutions) is a high-performance engineering platform designed to bridge the gap between intuitive visual design and rigorous mathematical analysis.

It enables engineers to model complex systems using a node-based interface, explore high-dimensional **Solution Spaces**, and optimize designs using industry-standard algorithms. Built for robustness, it features a crash-free multi-threaded architecture, vectorized computation kernels, and integrated AI capabilities.

---

## 🔬 Scientific Foundation: Solution Spaces

PyLCSS implements the **Solution Space** approach for robust design. Instead of seeking a single optimal point (which may be sensitive to manufacturing tolerances), PyLCSS identifies **box-shaped regions** of valid designs. This allows for decoupled development of subsystems in complex engineering projects.

> **Reference Algorithm:** > The solution space computation methods are based on:  
> *Markus Zimmermann, Johannes Edler von Hoessle*, "Computing solution spaces for robust design", *International Journal for Numerical Methods in Engineering*, 2013.  
> [DOI: 10.1002/nme.4450](https://doi.org/10.1002/nme.4450)

---

## ✨ Key Features

### 🏗️ Visual Modeling Environment
* **Node-Based Architecture:** Intuitive drag-and-drop interface powered by `NodeGraphQt`.
* **Unit Intelligence:** Automatic dimensional analysis and compatibility checking via `Pint` ensures physical consistency.
* **Python Integration:** Write custom logic blocks with full `NumPy` support.
* **CAD Modeling:** Parametric 3D CAD design using `CadQuery` with node-based workflow.

### 📊 Advanced Analysis Suite
* **Monte Carlo Exploration:** Rapidly evaluate thousands of design variants using vectorized sampling.
* **Solution Space Visualization:** Interactive 2D scatter plots, parallel coordinates, and feasibility maps.
* **Global Sensitivity Analysis:** Variance-based Sobol indices (via `SALib`) to identify critical design drivers.
* **FEM Simulation:** Finite element analysis with `scikit-fem` and `Netgen` meshing for structural analysis.

### 🧠 AI & Optimization
* **Surrogate Modeling:** Replace expensive simulations with fast approximations using **PyTorch** Neural Networks, Random Forests, or Gradient Boosting.
* **Multi-Objective Optimization:** Generate Pareto fronts using state-of-the-art solvers:
    * **Gradient-Based:** SLSQP (SciPy)
    * **Gradient-Free:** Nevergrad, Differential Evolution, COBYLA

### ⚡ Industrial-Grade Performance
* **Vectorized Kernels:** Calculation engines are optimized with NumPy vectorization for maximum throughput.
* **Non-Blocking UI:** Heavy computations run in background threads with signal throttling to ensure the GUI remains responsive at 60 FPS.
* **Crash Protection:** Robust error handling and race-condition prevention using Mutex locks.

---

## 📦 Installation

### Prerequisites
* **Python:** 3.8 or higher
* **OS:** Windows 10/11, macOS, or Linux

### Quick Install

```bash
# 1. Clone the repository
git clone <repository-url>
cd pylowcodesolutionspace

# 2. Create and activate a virtual environment (Recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch PyLCSS
python scripts/main.py
```

## 🚀 Quick Start Guide

**Launch the App:** Run `run_gui.bat` (Windows) or execute `python scripts/main.py`.

**Load a Model:** Navigate to `File > Open` and select `data/Gear Unit.json`.

**Validate:** Click the "Validate" button to check for unit consistency and connection errors.

**Compute:** Switch to the Solution Space tab and click "Compute" to generate design samples.

**Visualize:** Use the "Plot Settings" to visualize trade-offs between Weight vs Safety Factor.

**Optimize:** Go to the Optimization tab, select objectives (e.g., Minimize Weight), and run the solver.

## 📚 Tech Stack

PyLCSS is built on the shoulders of giants:

**UI/UX:** PySide6, NodeGraphQt, QtAwesome

**Computation:** NumPy, SciPy, Pandas

**Visualization:** PyQtGraph

**Machine Learning:** PyTorch, Scikit-learn

**Optimization:** Nevergrad, SALib

**Physics:** Pint

**CAD and FEM:** CadQuery, VTK, scikit-fem, Netgen, meshio

## 📄 License

PyLCSS is licensed under the **PolyForm Shield License 1.0.0**.

**✅ Allowed:** Personal use, academic research, internal business use.

**❌ Restricted:** You cannot use this software to build a competing product or service.

See [LICENSE](LICENSE) for full details.

<div align="center"> <sub>Copyright © 2025 Kutay Demir. All rights reserved.</sub> </div>
