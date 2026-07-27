# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Searchable in-app guide for the workflows exposed by the current UI."""

from PySide6 import QtCore, QtGui, QtWidgets


_HTML_HEAD = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body { background:#1e1f22; color:#dce2eb; font-family:'Segoe UI',sans-serif;
       font-size:13px; line-height:1.6; margin:20px 25px; }
h1 { color:#d9a441; font-size:21px; border-bottom:1px solid #343943;
     padding-bottom:8px; margin:0 0 12px; }
h2 { color:#e7bd63; font-size:15px; margin:22px 0 7px; }
p { margin:5px 0 10px; } ul,ol { margin:5px 0 12px 22px; padding:0; }
li { margin:3px 0; } b { color:#f2f5f9; }
code,pre { background:#15171b; border:1px solid #343943; border-radius:4px;
           font-family:'Consolas','Courier New',monospace; font-size:12px; }
code { padding:1px 5px; } pre { padding:10px 13px; white-space:pre-wrap; }
table { border-collapse:collapse; width:100%; margin:10px 0; }
th { background:#282c33; color:#e7bd63; text-align:left; }
th,td { border:1px solid #343943; padding:7px 9px; vertical-align:top; }
tr:nth-child(even) td { background:#1a1c20; } a { color:#65aaf2; }
.tip { background:#192844; border-left:3px solid #579bea; padding:9px 12px;
       border-radius:0 4px 4px 0; margin:12px 0; }
.warn { background:#322617; border-left:3px solid #d9a441; padding:9px 12px;
        border-radius:0 4px 4px 0; margin:12px 0; }
</style></head><body>"""


def _page(body: str) -> str:
    return _HTML_HEAD + body + "</body></html>"


_PAGES: dict[str, str] = {
"Start Here": _page("""
<h1>Start Here</h1>
<p>PyLCSS separates system-level design work from detailed CAD and simulation.
Choose the tab that matches the job.</p>
<table><tr><th>Task</th><th>Tab</th></tr>
<tr><td>Build a functional model from variables, calculations, and quantities of interest</td><td><b>Modeling Environment</b></td></tr>
<tr><td>Create geometry and run FEA, crash, or topology optimization</td><td><b>Design Studio</b></td></tr>
<tr><td>Train a fast approximation from model data or a saved CAD study</td><td><b>Surrogate Training</b></td></tr>
<tr><td>Sample feasible designs and compute solution spaces</td><td><b>Solution Space</b></td></tr>
<tr><td>Optimize objectives under constraints</td><td><b>Optimization</b></td></tr>
<tr><td>Rank variable influence on an output</td><td><b>Sensitivity Analysis</b></td></tr></table>
<h2>A reliable first workflow</h2><ol>
<li>In Modeling Environment, add Design Variable, Function Block, and QoI nodes.</li>
<li>Connect the ports and use <b>Validate</b>.</li>
<li>Use <b>Build Model</b> to forward the compiled model to Solution Space and Optimization.</li>
<li>Set QoI limits, objectives, and constraints in the target analysis tab.</li></ol>
<div class="tip">The robot button at the upper-right opens the optional AI Assistant. Its gear button opens provider settings.</div>
"""),
"Projects and Files": _page("""
<h1>Projects and Files</h1>
<h2>Application project folder</h2>
<p><b>File &rarr; Save Project</b> creates a folder named from the product and saves
Modeling, Surrogate, Solution Space, Optimization, and Sensitivity state.
<b>File &rarr; Load Project</b> loads such a folder.</p>
<h2>Design Studio file</h2>
<p>Design Studio has separate <b>New</b>, <b>Open</b>, and <b>Save</b> toolbar
commands. Its graph and settings are stored in a <code>.cad</code> file.</p>
<div class="warn"><b>Keep both when a project uses CAD:</b> the application project
folder does not replace the Design Studio <code>.cad</code> file.</div>
<h2>Computed results</h2>
<p><b>Save Results</b> in Design Studio exports an already-computed result as JSON
or HDF5; it does not run a solver. Geometry export uses the STEP or STL actions.</p>
"""),
"Interface": _page("""
<h1>Interface</h1>
<h2>Design Studio layout</h2><ul>
<li><b>Left:</b> searchable node library; double-click or drag an item.</li>
<li><b>Center top:</b> 3D viewer for previews and results.</li>
<li><b>Center bottom:</b> node graph.</li>
<li><b>Right top:</b> inspector for the selected node.</li>
<li><b>Right bottom:</b> Results and History tabs.</li></ul>
<p>FEA, Crash, and Topology setup lives in each solver node's inspector. There is
no separate Studies tab.</p>
<h2>Design Studio shortcuts</h2><table>
<tr><td>Run selected study or graph</td><td><code>F5</code></td></tr>
<tr><td>Stop computation</td><td><code>Shift+F5</code></td></tr>
<tr><td>Fit graph / Reset 3D view</td><td><code>F</code> / <code>R</code></td></tr>
<tr><td>Undo / Redo</td><td><code>Ctrl+Z</code> / <code>Ctrl+Y</code></td></tr>
<tr><td>Delete selected nodes</td><td><code>Delete</code></td></tr></table>
"""),
"Design Studio Overview": _page("""
<h1>Design Studio Overview</h1>
<p>Design Studio is a saved node graph for parametric geometry, preparation,
simulation, measurement, and export.</p>
<h2>Geometry sources</h2><ul>
<li>Native Box, Cylinder, Tube, shell, Boolean, hole, fillet, transform, and pattern nodes.</li>
<li><b>Code Part</b> for CadQuery scripts and named scalar parameters.</li>
<li><b>FreeCAD Part</b> for a node-owned FreeCAD document when FreeCAD is installed.</li>
<li>STEP/IGES/BREP or surface-mesh import.</li></ul>
<h2>Run behavior</h2>
<p>With exactly one node selected, <b>Run</b> evaluates that node and its upstream
dependencies. With no single selection, Run evaluates the graph.</p>
<div class="tip">Property edits perform a CAD-only preview. Expensive FEA, Crash,
and Topology solves start only from the explicit Run action.</div>
<h2>Solver branches</h2><ul>
<li><b>Static FEA:</b> Netgen mesh plus CalculiX.</li>
<li><b>Crash:</b> prepared mesh and OpenRadioss.</li>
<li><b>Topology:</b> direct design domain with structural and/or thermal conditions.</li></ul>
"""),
"Face Selection": _page("""
<h1>Face Selection</h1>
<p>Boundary conditions normally reference geometry through a <b>Pick Faces</b>
node. The selection is saved in the graph.</p><ol>
<li>Connect the geometry shape to Pick Faces.</li>
<li>Select it and click <b>Pick Faces in 3D Viewer</b>.</li>
<li>Pick and accept one or more faces.</li>
<li>Connect its <code>workplane</code> output to a condition face input.</li>
<li>Use <b>Preview in 3D</b> on FEA conditions to inspect the overlay.</li></ol>
<p>A selector node is also available for direction, point, index, box,
coordinate-expression, or tag selection.</p>
<div class="warn">Raw face indices can change with upstream geometry. Interactive
or geometric selectors are generally more stable.</div>
"""),
"Static FEA": _page("""
<h1>Static FEA</h1>
<p>The FEA Solver writes and runs a CalculiX static analysis. CalculiX is the
supported solve path.</p>
<h2>Required connections</h2><table>
<tr><th>Input</th><th>Connect</th></tr>
<tr><td><code>mesh</code></td><td>Mesh output using Tet or Tet10 solid elements</td></tr>
<tr><td><code>material</code></td><td>Material output</td></tr>
<tr><td><code>constraints</code></td><td>One or more Support outputs</td></tr>
<tr><td><code>loads</code></td><td>Force or Pressure; optional only for a nonzero prescribed displacement</td></tr></table>
<h2>Study Definition in the node</h2>
<p>Select the FEA Solver to see connection counts and CalculiX availability.
<b>Add Support</b>, <b>Add Force</b>, and <b>Add Pressure</b> create and connect a
condition. If the solver has a mesh, it is also connected to the new condition.
Connect a Pick Faces output to finish a face-based condition. Gravity needs no face.</p>
<h2>Settings</h2><ul>
<li>Linear, Nonlinear (Geometric), and Nonlinear (Plastic) analysis types.</li>
<li>Von Mises stress or displacement visualization with display-only deformation scale.</li>
<li><b>Deck only</b> writes the input without launching CalculiX.</li></ul>
<div class="warn">Static FEA accepts Tet and Tet10 solid meshes. Shell meshes are for the Crash Solver.</div>
"""),
"Crash": _page("""
<h1>Crash and Impact</h1>
<p>The Crash Solver prepares and runs an OpenRadioss explicit transient study.</p>
<h2>Required connections</h2><table>
<tr><th>Input</th><th>Connect</th></tr>
<tr><td><code>mesh</code></td><td>Mesh output; solid or shell as appropriate</td></tr>
<tr><td><code>crash_material</code></td><td>Crash Material output</td></tr>
<tr><td><code>impact</code></td><td>Impact Condition output</td></tr>
<tr><td><code>constraints</code></td><td>Support when the scenario requires a fixed specimen</td></tr></table>
<h2>Study Definition in the node</h2>
<p>Select the Crash Solver to see connection counts, impact scenario, and
OpenRadioss availability. <b>Add Impact</b> and <b>Add Support</b> create and wire
the corresponding nodes.</p>
<h2>Impact scenarios</h2><ul>
<li><b>Fixed specimen + moving impactor:</b> requires impact face and rear support.</li>
<li><b>Moving body + fixed wall:</b> requires neither impact face nor support.</li>
<li><b>Prescribed moving wall:</b> requires impact face but not support.</li></ul>
<p>Velocity is mm/ms (numerically m/s); End Time is ms. Mass scaling trades
physical inertia accuracy for a larger stable time step.</p>
<p><b>OpenRadioss Deck</b> runs an existing deck and does not use the graph-built Study Definition.</p>
"""),
"Topology Optimization": _page("""
<h1>Topology Optimization</h1>
<p>The Topology Solver works directly from a CAD design domain. Do not insert the
standard FEA Mesh node before it.</p>
<h2>Study Definition in the node</h2>
<p>The embedded section reports design domain, material, structural, multibody,
and thermal inputs. Buttons add Topology Support, Force, Joint, Operating Case,
Temperature Boundary, and Heat Input nodes.</p>
<h2>Common setups</h2><ul>
<li><b>Structural:</b> domain + material + topology support + topology force.</li>
<li><b>Multibody:</b> domain + material + joints and operating cases.</li>
<li><b>Thermal:</b> domain + material + temperature boundary + heat input.</li>
<li><b>Thermo-mechanical:</b> structural and thermal conditions.</li></ul>
<p>The same inspector owns goal, material budget, formulation, manufacturing,
optional CalculiX validation, CAD reconstruction, and visualization.</p>
<p>After a run, export the recovered surface to STL or STEP. Volume Remesh makes
a volume mesh for downstream validation.</p>
"""),
"Units and Solvers": _page("""
<h1>Units and External Solvers</h1><table>
<tr><th>Quantity</th><th>Unit</th></tr>
<tr><td>Length and displacement</td><td>mm</td></tr>
<tr><td>Force</td><td>N</td></tr>
<tr><td>Stress, pressure, Young's modulus</td><td>MPa = N/mm<sup>2</sup></td></tr>
<tr><td>Density</td><td>tonne/mm<sup>3</sup>; steel about <code>7.85e-9</code></td></tr>
<tr><td>Crash time / velocity</td><td>ms / mm/ms</td></tr></table>
<h2>Optional installations</h2><pre>python scripts/install_solvers.py --only ccx
python scripts/install_solvers.py --only radioss
python scripts/install_solvers.py --only freecad</pre>
<p>PyLCSS starts without them. Solver nodes can prepare decks, but a full solve
requires the executable. Library tooltips and Study Definition show status.</p>
"""),
"Modeling Environment": _page("""
<h1>Modeling Environment</h1><table>
<tr><th>Node</th><th>Role</th></tr>
<tr><td>Design Variable</td><td>Input with value, bounds, and unit</td></tr>
<tr><td>Intermediate</td><td>Named value between calculations</td></tr>
<tr><td>Function Block</td><td>Python calculation or linked Design Studio solver</td></tr>
<tr><td>QoI</td><td>Result, requirement, constraint, or objective</td></tr></table>
<h2>Workflow</h2><ol>
<li>Create or select a system.</li><li>Add nodes and connect ports.</li>
<li>Double-click nodes to edit configuration.</li><li>Use <b>Validate</b>.</li>
<li>Use <b>Build Model</b> to forward it to Solution Space and Optimization.</li></ol>
<p>A Design Studio Function Block reads a saved <code>.cad</code> graph, maps
inputs to exposed parameters, and calls the chosen FEA, Crash, or Topology terminal.</p>
"""),
"Surrogate Training": _page("""
<h1>Surrogate Training</h1><ol>
<li>Select a Function Block from Modeling Environment.</li>
<li>Generate samples or upload CSV/JSON.</li>
<li>Choose an available algorithm and configure it.</li>
<li>Train and inspect metrics, parity, learning curves, cross-validation, and feature importance.</li>
<li>Save only when error is acceptable for the intended decision.</li></ol>
<p>Core choices are MLP Regressor, Random Forest, Gradient Boosting, and Gaussian
Process. PyTorch and geometry-aware choices appear only with their dependencies.</p>
<div class="warn">Debug overfit modes intentionally train and test on the same tiny
data. They test the pipeline, not predictive accuracy.</div>
"""),
"Solution Space": _page("""
<h1>Solution Space</h1>
<p>This tab samples a compiled model and finds regions satisfying all QoI limits.</p><ol>
<li>Build a model.</li><li>Review variables and QoIs.</li>
<li>Define lower and/or upper QoI requirements.</li>
<li>Choose sample count and compute.</li>
<li>Inspect plots, data, and feasible box; export when needed.</li></ol>
<p><b>Multi-Modal</b> searches separated feasible regions. <b>Product Family</b>
compares variants and common-platform choices.</p>
<p>Method background: <a href="https://doi.org/10.1002/nme.4450">Computing solution spaces for robust design</a>.</p>
"""),
"Optimization": _page("""
<h1>Optimization</h1><ol>
<li>Select a compiled system.</li><li>Choose QoI objectives and direction.</li>
<li>Configure constraints and variable bounds.</li>
<li>Choose an algorithm and settings.</li>
<li>Run and inspect objective, variable, constraint, and Pareto plots.</li></ol>
<table><tr><th>Algorithm</th><th>Typical use</th></tr>
<tr><td>SLSQP</td><td>Smooth constrained local problems</td></tr>
<tr><td>COBYLA / trust-constr</td><td>Derivative-free or robust constrained local search</td></tr>
<tr><td>Differential Evolution / Nevergrad</td><td>Global or black-box search</td></tr>
<tr><td>NSGA-II</td><td>Multiple objectives and Pareto front</td></tr>
<tr><td>Multi-Start</td><td>Repeated local search from multiple starts</td></tr></table>
"""),
"Sensitivity Analysis": _page("""
<h1>Sensitivity Analysis</h1><ol>
<li>Build the system model.</li><li>Select method and output.</li>
<li>Set base sample size or Morris trajectories.</li>
<li>Run and inspect method-specific plots and tables.</li></ol>
<table><tr><th>Method</th><th>Behavior</th></tr>
<tr><td>Sobol</td><td>First, total, and optional second-order variance effects</td></tr>
<tr><td>Morris</td><td>Low-cost elementary-effect screening</td></tr>
<tr><td>FAST</td><td>Fourier-based first-order sensitivity</td></tr>
<tr><td>Delta</td><td>Moment-independent distributional sensitivity</td></tr></table>
<p>Available methods depend on installed dependencies. Use the UI's evaluation
estimate for the selected method and variable count.</p>
"""),
"AI Assistant": _page("""
<h1>AI Assistant</h1>
<p>Click the upper-right robot button, enter a request, and press Enter or Send.
The assistant can call registered in-app actions.</p>
<h2>Setup</h2><ol><li>Open the panel and click the gear.</li>
<li>Select a cloud provider or OpenAI-compatible local server.</li>
<li>Enter credentials or base URL, select a model, and test.</li>
<li>Save and begin with a small request.</li></ol>
<p>State the target tab, dimensions, units, bounds, and solver type. Ask for graph
validation after node changes.</p>
<div class="warn">Review generated engineering work. Validate units, connections,
boundary conditions, solver settings, and results before decisions.</div>
"""),
"Troubleshooting": _page("""
<h1>Troubleshooting</h1>
<h2>Solver will not start</h2><ul>
<li>Check the backend row in the solver's Study Definition.</li>
<li>Hover the solver in the library for detection detail.</li>
<li>Use Deck Only if you only need an input deck.</li></ul>
<h2>Connected study still fails</h2><ul>
<li>Give each FEA condition its required mesh and face.</li>
<li>For FEA, provide support and load unless displacement drives the model.</li>
<li>For fixed-specimen crash, provide impact face and support.</li></ul>
<h2>Wrong branch runs</h2><p>Select exactly the terminal node, then press F5.</p>
<h2>3D viewer is empty</h2><p>Run a geometry node first. Face picking also requires
an upstream shape and a completed preview.</p>
"""),
"About": _page("""
<h1>About PyLCSS</h1>
<p><b>PyLCSS 2.2.0</b> is a source-available desktop environment for system
modeling, parametric CAD, simulation, solution-space exploration, optimization,
sensitivity analysis, and surrogate modeling.</p>
<p>Copyright &copy; 2026 Kutay Demir.</p>
<p>Licensed under the <b>PolyForm Shield License 1.0.0</b>. See LICENSE and NOTICE.</p>
"""),
}

_NAV = [
    ("Getting Started", ["Start Here", "Projects and Files", "Interface"]),
    ("Design Studio", ["Design Studio Overview", "Face Selection", "Static FEA",
                       "Crash", "Topology Optimization", "Units and Solvers"]),
    ("System Workflows", ["Modeling Environment", "Surrogate Training",
                          "Solution Space", "Optimization", "Sensitivity Analysis"]),
    ("Support", ["AI Assistant", "Troubleshooting", "About"]),
]


class HelpWidget(QtWidgets.QWidget):
    """Two-panel help browser with title and content search."""

    _SIDEBAR_STYLE = """
        QWidget#helpSidebar { background:#1a1c20; }
        QTreeWidget { background:#1a1c20; border:none; outline:0; font-size:12px; }
        QTreeWidget::item { padding:4px 6px; color:#b9c0ca; }
        QTreeWidget::item:selected { background:#2c3546; color:white; }
        QTreeWidget::item:hover:!selected { background:#24272d; color:#eef2f7; }
        QLineEdit { background:#24272d; border:1px solid #363b44; border-radius:5px;
                    color:#e1e6ed; padding:6px 8px; }
        QLineEdit:focus { border-color:#579bea; }
    """

    def __init__(self) -> None:
        super().__init__()
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("helpSidebar")
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet(self._SIDEBAR_STYLE)
        side_layout = QtWidgets.QVBoxLayout(sidebar)
        side_layout.setContentsMargins(9, 11, 9, 9)
        side_layout.setSpacing(7)
        title = QtWidgets.QLabel("PyLCSS Help")
        title.setStyleSheet("color:#d9a441; font-size:15px; font-weight:700; padding:2px 4px 5px;")
        side_layout.addWidget(title)

        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("Search help")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter_tree)
        side_layout.addWidget(self._search)
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(14)
        self._tree.setUniformRowHeights(True)
        self._tree.itemClicked.connect(self._on_tree_click)
        side_layout.addWidget(self._tree, 1)

        self._browser = QtWidgets.QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._browser.setStyleSheet("background:#1e1f22; border:none;")
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setStyleSheet("color:#343943;")
        divider.setFixedWidth(1)
        root.addWidget(sidebar)
        root.addWidget(divider)
        root.addWidget(self._browser, 1)

        self._build_tree()
        first = self._tree.topLevelItem(0).child(0)
        self._tree.setCurrentItem(first)
        self._show_page(first.data(0, QtCore.Qt.UserRole))

    def _build_tree(self) -> None:
        self._tree.clear()
        for section, pages in _NAV:
            parent = QtWidgets.QTreeWidgetItem([section])
            parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsSelectable)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setForeground(0, QtGui.QColor("#82b8ef"))
            self._tree.addTopLevelItem(parent)
            for page_key in pages:
                item = QtWidgets.QTreeWidgetItem([page_key])
                item.setData(0, QtCore.Qt.UserRole, page_key)
                parent.addChild(item)
            parent.setExpanded(True)

    def _filter_tree(self, query: str) -> None:
        query = query.strip().casefold()
        for index in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(index)
            visible_children = 0
            for child_index in range(parent.childCount()):
                child = parent.child(child_index)
                key = str(child.data(0, QtCore.Qt.UserRole) or "")
                visible = not query or query in (key + " " + _PAGES.get(key, "")).casefold()
                child.setHidden(not visible)
                visible_children += int(visible)
            parent.setHidden(bool(query) and not visible_children)
            if query and visible_children:
                parent.setExpanded(True)

    def _on_tree_click(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        key = item.data(0, QtCore.Qt.UserRole)
        if key:
            self._show_page(str(key))

    def _show_page(self, key: str) -> None:
        self._browser.setHtml(_PAGES.get(key, _page(f"<h1>{key}</h1><p>Page not found.</p>")))
        self._browser.verticalScrollBar().setValue(0)
