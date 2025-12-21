# PyLCSS: Low-Code System Solutions

<div align="center">

**🚀 Source-Available Engineering Simulation Platform**

*Visual modeling, optimization, and AI-powered analysis for engineers*

</div>

---

## 🎯 What is PyLCSS?

PyLCSS (Python Low-Code System Solutions) is a comprehensive source-available platform designed to accelerate engineering analysis and design optimization. Built specifically for engineers, it combines an intuitive visual programming interface with advanced computational algorithms to transform complex engineering problems into solvable optimization challenges.

**Current Version: 1.3.0** - Complete engineering analysis suite with multi-objective optimization, global sensitivity analysis, AI-powered surrogate modeling, and advanced constraint handling.

### ✨ Key Benefits

- **🏗️ Visual System Design**: Drag-and-drop interface for building complex engineering systems
- **📊 Advanced Analytics**: Multi-objective optimization, sensitivity analysis, uncertainty quantification, and solution space exploration
- **🧠 AI Integration**: Machine learning surrogate models with PyTorch support for computationally intensive simulations
- **⚡ Performance Optimized**: Parallel processing, efficient algorithms, and smart constraint scaling for large-scale problems
- **🛡️ Production Ready**: Robust error handling, comprehensive validation, and user-friendly interface

---

## 🚀 Core Features

### 🔬 Advanced Analysis Suite

| Feature | Description | Applications |
|---------|-------------|--------------|
| **Multi-Objective Optimization** | Pareto-optimal solutions with multiple conflicting objectives using SLSQP, Nevergrad, and Differential Evolution | Design trade-off analysis, cost-performance optimization |
| **Global Sensitivity Analysis** | Sobol variance-based sensitivity with confidence intervals | Parameter importance ranking, uncertainty propagation |
| **Surrogate Modeling** | ML approximations including PyTorch neural networks for expensive simulations | Computational cost reduction, real-time analysis |
| **Solution Space Exploration (Markus Zimmermann, Johannes Edler von Hoessle, Computing solution spaces for robust design https://doi.org/10.1002/nme.4450)** | Monte Carlo sampling with interactive visualization and optional optimization inclusion | Design space characterization, feasibility analysis |
| **Constraint Scaling** | Automatic scaling of variables and constraints to improve solver convergence | Robust optimization of problems with mixed scales |

### 🏗️ System Modeling Engine

- **🎨 Node-Based Architecture**: Visual programming with drag-and-drop components
- **🐍 Python Integration**: Custom function blocks with full Python syntax and NumPy support
- **📏 Unit Management**: Automatic dimensional analysis and unit conversion with Pint
- **✅ Real-Time Validation**: Immediate feedback on model connectivity and errors
- **🔄 System Merging**: Combine multiple sub-systems for complex analysis

### 📈 Analysis Capabilities

- **🎯 Optimization Algorithms**: SLSQP, COBYLA, Nevergrad, Differential Evolution with native constraint support
- **📊 Statistical Analysis**: Monte Carlo sampling, confidence intervals, convergence metrics
- **🔍 Interactive Visualization**: Real-time plotting, data filtering, Pareto front exploration
- **💾 Data Management**: HDF5-based storage, CSV export, project persistence
- **⚖️ Smart Scaling**: Automatic variable and constraint scaling for improved convergence

---

## 📚 Documentation

**Integrated Help System**: PyLCSS features a comprehensive built-in documentation system directly within the GUI. 

- **Context-Aware Help**: Hover over any component or setting to see detailed tooltips.
- **Help Widget**: Access the dedicated Help tab in the application for full guides on:
    - Getting Started
    - Node Reference
    - Optimization Strategies
    - Scripting API
- **Examples**: The `data/` directory contains example projects (e.g., `Gear Unit.json`) to help you get started immediately.

---

## 📦 Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Operating System** | Windows 10 / Ubuntu 18.04 / macOS 10.15 | Windows 11 / Ubuntu 22.04 / macOS 12+ |
| **Python Version** | 3.8 | 3.11+ |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 2 GB | 5 GB+ |
| **Display** | 1920x1080 | 2560x1440+ |

### Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd pylcss

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install PyLCSS
pip install -e .

# Launch the application
pylcss
```

### 🏃‍♂️ Quick Start

1.  **Launch PyLCSS**: Run `pylcss` (or `run_gui.bat` on Windows).
2.  **Load Example**: Go to `File > Open` and select `data/Gear Unit.json`.
3.  **Explore**:
    *   **Editor Tab**: View the node graph structure.
    *   **Solution Space Tab**: Click "Compute Solution Space" to visualize feasible designs.
    *   **Optimization Tab**: Run multi-objective optimization to find Pareto fronts.

---

## 📄 License

This project is licensed under the **PolyForm Shield License 1.0.0**. See the [LICENSE](LICENSE) file for details.

*Free for non-competing use.*
