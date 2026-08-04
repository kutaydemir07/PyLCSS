# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Searchable in-app guide for the workflows exposed by the current UI."""

from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.user_interface.common.theme_manager import THEMES, current_theme


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
<tr><td>Create geometry and run FEA, impact, or topology optimization</td><td><b>Design Studio</b></td></tr>
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
Modeling, Design Studio, Surrogate, Solution Space, Optimization, and
Sensitivity state. Design Studio simulation results are compressed into an
HDF5 sidecar. <b>File &rarr; Load Project</b> restores the complete folder.</p>
<h2>Design Studio file</h2>
<p>Design Studio has separate <b>New</b>, <b>Open</b>, and <b>Save</b> toolbar
commands. Its graph and settings are stored in a <code>.cad</code> file; cached
FEA, impact, topology, and remesh results are stored beside it in
<code>.cad.results.h5</code>. Keep both files when sharing a standalone study.</p>
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
<li><b>Right bottom:</b> Results for the selected node.</li></ul>
<p>FEA, Impact, and Topology studies are built by connecting nodes in the graph;
the inspector holds each node's settings. There is no separate Studies tab.</p>
<h2>Design Studio shortcuts</h2><table>
<tr><td>Run selected study or graph</td><td><code>F5</code></td></tr>
<tr><td>Stop computation</td><td><code>Shift+F5</code></td></tr>
<tr><td>Fit graph / Reset 3D view</td><td><code>F</code> / <code>R</code></td></tr>
<tr><td>Undo / Redo</td><td><code>Ctrl+Z</code> / <code>Ctrl+Y</code></td></tr>
<tr><td>Delete selected nodes</td><td><code>Delete</code></td></tr></table>
<p>Choose <b>View &rarr; Theme</b> for the complete dark or light interface.</p>
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
dependencies. With no selection, one detected solver workflow runs directly;
when several solver terminals exist, select the FEA, impact, or topology terminal
to choose the workflow. Sibling workflows never run implicitly.</p>
<div class="tip">Property edits perform a CAD-only preview. Expensive FEA, Impact,
and Topology solves start only from the explicit Run action.</div>
<h2>Solver branches</h2><ul>
<li><b>Static FEA:</b> Netgen tetrahedral mesh plus CalculiX.</li>
<li><b>Impact:</b> prepared mesh and OpenRadioss.</li>
<li><b>Topology:</b> direct design domain with structural and/or thermal conditions.</li></ul>
"""),
    "Geometry Selection": _page("""
<h1>Faces, Edges, and Vertices</h1>
<p>Boundary conditions reference CAD through <b>Select Geometry</b> or
<b>Pick Geometry</b>. Choose Face, Edge, or Vertex in the node;
the selection and geometric summaries are saved in the graph.</p><ol>
<li>Connect the upstream CAD shape to Select Geometry.</li>
<li>Choose the entity type. For the interactive node, click its viewer-pick button.</li>
<li>Select one or more entities and accept the selection.</li>
<li>Connect the selector output to a support, force, pressure, impact, or topology condition.</li>
<li>Select any condition or solver node to inspect the persistent 3D overlay.</li></ol>
<p>Supports and concentrated loads accept faces, edges, or vertices. Pressure
requires an area and therefore accepts faces only. Mesh vertices can be selected
geometrically; CAD edges should be selected before remeshing because an
unstructured mesh does not preserve stable CAD-edge identity.</p>
<div class="warn">Raw entity indices can change when upstream topology changes.
Direction, nearest-point, box, tag, or interactive geometric selection is more stable.</div>
"""),
    "Meshing": _page("""
<h1>Meshing</h1>
<p><b>Netgen</b> imports STEP through OpenCASCADE and generates solid tetrahedral
or shell-surface meshes. Local sizing is supported on selected CAD faces.
Tet10 converts the linear volume mesh to quadratic C3D10 connectivity.</p>
<table><tr><th>Mesh</th><th>Use</th></tr>
<tr><td>Tet (C3D4)</td><td>Fast preliminary solid studies; refine and check convergence.</td></tr>
<tr><td>Tet10 (C3D10)</td><td>Preferred for stress and bending in static solid FEA.</td></tr>
<tr><td>Shell triangles</td><td>OpenRadioss thin-wall models; the CAD input must already be the intended midsurface.</td></tr></table>
<p>The node rejects collapsed elements and reports mean-ratio quality. Passing
that check is not proof of accuracy: inspect poor-element locations and perform
a mesh-convergence study on decision quantities.</p>
<div class="warn">Meshing the boundary of a thick solid as Shell does not extract
a physical midsurface and can duplicate skins or cap openings. Use a surface
geometry such as Cylindrical Shell or an imported/prepared midsurface.</div>
"""),
    "Static FEA": _page("""
<h1>Static FEA</h1>
<p>Static Solver runs CalculiX.</p>
<h2>Required connections</h2><table>
<tr><th>Input</th><th>Connect</th></tr>
<tr><td><code>mesh</code></td><td>Mesh output using Tet or Tet10 solid elements</td></tr>
<tr><td><code>material</code></td><td>Material output</td></tr>
<tr><td><code>components</code></td><td>Body outputs for assemblies</td></tr>
<tr><td><code>constraints</code></td><td>One or more Support outputs</td></tr>
<tr><td><code>loads</code></td><td>Force or Pressure; optional only for a nonzero prescribed displacement</td></tr></table>
<h2>Building the study</h2>
<p>Add Body, Support, Force, and Pressure nodes from the node library and connect
them to the solver inputs above. If the solver has a mesh, connect it to the
condition as well. Connect Select Geometry or Pick Geometry to finish the
condition. Supports and total forces can use a face, edge, or vertex; pressure
requires a face. Gravity needs no geometry selection. CalculiX availability is
shown in the node library.</p>
<h2>Settings</h2><ul>
<li>Linear, Nonlinear (Geometric), and Nonlinear (Plastic) analysis types.</li>
<li>Von Mises stress or displacement visualization with display-only deformation scale.</li>
<li><b>Deck only</b> writes the input without launching CalculiX.</li></ul>
<h2>Implemented CalculiX subset</h2>
<p>PyLCSS currently validates 3D C3D4/C3D10 static solids with isotropic
elasticity, optional bilinear plasticity, geometric nonlinearity, translational
supports/prescribed displacement, nodal force, face pressure, and gravity.
It imports displacement and stress from FRD results.</p>
<div class="warn">This is not the complete CalculiX feature set. Shells, beams,
contact, modal, buckling, transient dynamics, thermal coupling, composites,
pretension, cyclic symmetry, and user materials are not graph-built FEA
features. Use an external deck/tool for those studies. Always check reactions,
units, mesh convergence, and solver logs.</div>
"""),
    "Impact": _page("""
<h1>Impact and Crashworthiness</h1>
<p>Impact Solver runs an OpenRadioss transient study.</p>
<h2>Required connections</h2><table>
<tr><th>Input</th><th>Connect</th></tr>
<tr><td><code>mesh</code></td><td>Mesh output; solid or shell as appropriate</td></tr>
<tr><td><code>impact_material</code></td><td>Impact Material output</td></tr>
<tr><td><code>impact</code></td><td>Impact Setup output</td></tr>
<tr><td><code>constraints</code></td><td>Support when the scenario requires a fixed specimen</td></tr></table>
<h2>Building the study</h2>
<p>Add Impact Setup and Support nodes from the node library and connect them to
the solver inputs above. OpenRadioss availability is shown in the node
library.</p>
<h2>Impact scenarios</h2><ul>
<li><b>Fixed specimen + moving impactor:</b> requires impact face and rear support.</li>
<li><b>Moving body + fixed wall:</b> requires neither impact face nor support.</li>
<li><b>Prescribed moving wall:</b> requires impact face but not support.</li></ul>
<p>Velocity is mm/ms (numerically m/s); impactor/sled mass is defined on the
Impact Setup; End Time is ms. Mass scaling trades
physical inertia accuracy for a larger stable time step.</p>
<h2>Results</h2>
<p>Choose the displayed field and the deformation scale on the Impact Solver
itself (<b>Result Display</b> and <b>Deformation Scale</b>); switching either
one redraws the cached result without solving again. Peak crushing force,
acceleration, specific energy absorption, mass, and energy-balance error are
listed in the Results panel after a run.</p>
<h2>Implemented OpenRadioss subset</h2>
<p>The graph builder covers one shell or Tet4 part with isotropic
elastic-plastic kinematic material, optional Cowper-Symonds rate response and
calibrated element deletion, translational SPC, initial velocity or planar
rigid-wall impact, automatic single-surface contact, and global energy/mass
history checks.</p>
<div class="warn">Material presets are starting values, not validated production
cards. Calibrate the actual alloy, thickness, rate, hardening, and failure law.
The graph builder does not cover general multi-part interfaces, spotwelds,
composites, airbags, ALE/SPH/CFD, occupant models, or arbitrary rigid bodies.
Use OpenRadioss Deck for those models. Treat energy creation above 2%, final
energy loss beyond roughly 15%, or mass change above 1% as investigation flags,
not automatic proof of validity.</div>
<p><b>OpenRadioss Deck</b> runs an existing deck and does not use the graph-built study.</p>
"""),
    "Topology Optimization": _page("""
<h1>Topology Optimization</h1>
<p>The Topology Solver works directly from a CAD design domain. Do not insert the
standard FEA Mesh node before it.</p>
<h2>Building the study</h2>
<p>Connect design domain, material, and the structural, multibody, or thermal
inputs on the node. The connected <b>Design domain</b> is the optimizable source
body. Add Fixed Support, Force, Passive Region, Joint, Load Case, Heat Sink, and
Heat Load nodes from the node library. A Passive Region takes a closed CAD solid
and preserves it as material or void.</p>
<h2>Common setups</h2><ul>
<li><b>Structural:</b> domain + material + topology support + topology force.</li>
<li><b>Multibody:</b> domain + material + joints and load cases.</li>
<li><b>Thermal:</b> domain + material + temperature boundary + heat input.</li>
<li><b>Thermo-mechanical:</b> structural and thermal conditions.</li></ul>
<p>The same inspector owns goal, material budget, formulation, manufacturing,
optional CalculiX validation, CAD reconstruction, and visualization.</p>
<p>The result is a solid envelope: it is reconstructed automatically after a
run, STEP exports that CAD body directly, and STL tessellates the same body, so
both formats represent the same result. Volume Remesh makes a volume mesh for
downstream validation.</p>
<p>For a graded cellular result, use the <b>Lattice Solver</b> instead. It is a
separate node because it is a separate design problem, not a different way of
displaying this one.</p>
"""),
    "Lattice Optimization": _page("""
<h1>Lattice Optimization</h1>
<p>The <b>Lattice Solver</b> takes the same study definition as the Topology
Solver &mdash; design domain, material, supports, forces, passive regions,
joints, load cases and thermal conditions &mdash; and optimizes a
variable-density cellular structure instead of a solid envelope.</p>
<h2>Why it is its own node</h2>
<p>SIMP penalization and the Heaviside projection exist to drive the density
field to fully solid or fully empty. A graded lattice <i>is</i> the intermediate
density, so the two work against each other: a penalized, projected field
reaches full density throughout the structure, every voxel clears the
solid-transition threshold, and no lattice is generated at all. The lattice
study therefore solves the selected cell's homogenized law and keeps the graded
field, which is what sizes the explicit cells.</p>
<h2>Cell families</h2>
<ul>
<li><b>Sheet TPMS</b> &mdash; Gyroid is the general-purpose default and Schwarz
Primitive provides straight channels for flow and heat-transfer studies.</li>
<li><b>Strut lattices</b> &mdash; BCC is a common compliant metal-AM cell and
Octet Truss is the stretch-dominated structural reference. These can have
their member diameters sized against axial stress and buckling.</li>
<li><b>Honeycomb</b> &mdash; an extruded prismatic cell for strongly directional
stiffness.</li>
</ul>
<p>The new-study catalogue is intentionally limited to these five established
choices. Diamond and other specialist cells from earlier releases remain loadable for saved
project compatibility. Most public families are two-scale: the part is optimized against that cell's
measured homogenized anisotropic stiffness, then the density field is converted
back into explicit cells. Honeycomb has no cubic homogenized tensor and keeps
the isotropic continuum surrogate; the study report says which law was used.
Cell orientation is fixed to the global axes. Printability, drainage and the
manufactured geometry must be validated in every case.</p>
<h2>Setting the cell</h2>
<p><b>Guided</b> derives a family-safe cell pitch, wall or member thickness,
skin and density grading from the analysis grid. <b>Manual</b> exposes those
dimensions in model units, which is what a printer capability &mdash; minimum
feature size, powder evacuation &mdash; is actually stated in, together with a
target relative density: the one number that says what the part will weigh, and
the one that cell pitch and member thickness do not predict between them.</p>
<h2>Results and export</h2>
<p>A lattice is delivered as an explicit manufactured mesh and exports as STL.
BCC and Octet additionally have an exact analytic B-rep built from their
centrelines &mdash; one quadric per member, one ball per node &mdash; so they
also export as STEP, and as a <b>3MF beam lattice</b>: nodes and beams with
per-end radii, which the AM tool chain reads natively and which is a few
hundred kilobytes where the same body as STEP is megabytes. A minimal-surface
or honeycomb cell has no compact boundary representation and stays a mesh.</p>
<p>The built-in voxel check does not resolve member bending, joint stress,
fatigue or process defects. Validate the explicit manufactured geometry with an
appropriate beam or solid model before engineering release.</p>
"""),
    "Units and Solvers": _page("""
<h1>Units and External Solvers</h1><table>
<tr><th>Quantity</th><th>Unit</th></tr>
<tr><td>Length and displacement</td><td>mm</td></tr>
<tr><td>Force</td><td>N</td></tr>
<tr><td>Stress, pressure, Young's modulus</td><td>MPa = N/mm<sup>2</sup></td></tr>
<tr><td>Density</td><td>tonne/mm<sup>3</sup>; steel about <code>7.85e-9</code></td></tr>
<tr><td>Impact time / velocity</td><td>ms / mm/ms</td></tr></table>
<h2>Optional installations</h2><pre>python scripts/install_solvers.py --only ccx
python scripts/install_solvers.py --only radioss
python scripts/install_solvers.py --only freecad</pre>
<p>PyLCSS starts without them. Solver nodes can prepare decks, but a full solve
requires the executable. Library tooltips show detection status.</p>
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
inputs to exposed parameters, and calls the chosen FEA, Impact, or Topology terminal.</p>
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
<tr><td>COBYLA</td><td>Derivative-free local search with inequality constraints</td></tr>
<tr><td>trust-constr</td><td>Smooth nonlinear constrained problems</td></tr>
<tr><td>Differential Evolution / Nevergrad</td><td>Bounded stochastic or black-box search</td></tr>
<tr><td>NSGA-II</td><td>Multiple objectives and a sampled Pareto-front approximation</td></tr>
<tr><td>Multi-Start</td><td>Repeated local search from multiple starts</td></tr></table>
<p>For scalar multi-objective runs, each objective is divided by an explicit
reference scale or by its frozen initial-design magnitude before weights are
applied. This prevents kilograms, pascals, and millimetres from competing only
because of unit size. Stochastic methods expose a reproducible random seed.
NSGA-II uses Pareto dominance; its displayed compromise is the normalized
utopia-distance point and is not an automatically preferred engineering design.</p>
<div class="warn">Optimization does not validate the model. Re-evaluate the final
design, constraints, neighbouring points, and—when stochastic—more than one seed.</div>
"""),
    "Sensitivity Analysis": _page("""
<h1>Sensitivity Analysis</h1><ol>
<li>Build the system model.</li><li>Select method and output.</li>
<li>Set base sample size or Morris trajectories.</li>
<li>Run and inspect method-specific plots and tables.</li></ol>
<table><tr><th>Method</th><th>Behavior</th></tr>
<tr><td>Sobol</td><td>First, total, and optional second-order variance effects</td></tr>
<tr><td>Morris</td><td>Low-cost elementary-effect screening</td></tr>
<tr><td>eFAST</td><td>Fourier-based first-order and total-order effects</td></tr>
<tr><td>Delta</td><td>Moment-independent distributional sensitivity</td></tr></table>
<p>Sobol uses scrambled Saltelli-compatible sampling and a power-of-two base
size; second-order interactions are optional. Morris exposes trajectories and
an even grid level. FAST enforces its minimum sample requirement. Every
stochastic method exposes a seed and rejects non-finite or constant responses.</p>
<p>Available methods depend on installed dependencies. Use the UI's exact
evaluation estimate for the selected method and variable count. Sensitivity
indices describe the chosen input distributions and bounds, not universal
causality; repeat or inspect confidence intervals before decisions.</p>
"""),
    "AI Assistant": _page("""
<h1>AI Assistant</h1>
<p>Click the upper-right robot button, enter a request, and press Enter or Send.
The assistant can inspect Design Studio state; create and connect native CAD,
condition, solver, and expert code nodes; set properties; run one explicitly
selected workflow; stop it; and export cached CAD or recovered topology
geometry. Long solver runs stay in the application's background worker.</p>
<h2>Setup</h2><ol><li>Open the panel and click the gear.</li>
<li>Select a cloud provider or OpenAI-compatible local server.</li>
<li>Enter credentials or base URL, select a model, and test.</li>
<li>Save and begin with a small request.</li></ol>
<h2>Data and privacy</h2>
<p>The Local provider sends requests only to the configured local endpoint.
Cloud providers receive the current request, recent history when memory is
enabled, and a snapshot of the active graph including node properties and
connections. Provider terms and privacy policies apply. Conversation memory is
off by default and, when enabled, is stored locally in plaintext. Loading a
cloud model list sends the API key only after you explicitly press Load Models
or Test Connection.</p>
<p>State the target tab, dimensions, units, bounds, and solver type. Ask for graph
validation after node changes. If a graph has multiple solver terminals, name
the exact terminal to run.</p>
<div class="warn">Review generated engineering work. Validate units, connections,
boundary conditions, solver settings, and results before decisions. Do not send
confidential or personal data to a cloud provider unless authorized.</div>
"""),
    "Security and Trusted Files": _page("""
<h1>Security and Trusted Files</h1>
<p>Design Studio projects can include CadQuery Code nodes. Their Python executes
with your user permissions when the graph is previewed or solved. PyLCSS asks
for trust before executing code from a newly opened external project.</p>
<p>Serialized joblib, pickle, dill, and PyTorch model files can also execute code
during loading. Open projects, scripts, model files, plugins, and solver decks
only from sources you trust. Inspect Code nodes before approving execution.</p>
<p>The AI assistant can modify in-application graphs and launch selected PyLCSS
workflows, but it has no system-wide mouse or keyboard automation capability.
Keep automatic tool execution disabled unless the project, model, and request
are trusted.</p>
<div class="warn">A file extension is not a security boundary. Do not open an
unknown .cad or serialized model merely because it looks like ordinary data.</div>
"""),
    "Troubleshooting": _page("""
<h1>Troubleshooting</h1>
<h2>Solver will not start</h2><ul>
<li>Hover the solver in the library for detection detail.</li>
<li>Use Deck Only if you only need an input deck.</li></ul>
<h2>Connected study still fails</h2><ul>
<li>Give each FEA condition its required mesh and geometry selection.</li>
<li>For FEA, provide support and load unless displacement drives the model.</li>
<li>For a fixed-specimen impact, provide impact face and support.</li></ul>
<h2>Wrong branch runs</h2><p>Select exactly the terminal node, then press F5.</p>
<h2>3D viewer is empty</h2><p>Run a geometry node first. Entity picking also requires
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
    (
        "Design Studio",
        [
            "Design Studio Overview",
            "Geometry Selection",
            "Meshing",
            "Static FEA",
            "Impact",
            "Topology Optimization",
            "Lattice Optimization",
            "Units and Solvers",
        ],
    ),
    (
        "System Workflows",
        [
            "Modeling Environment",
            "Surrogate Training",
            "Solution Space",
            "Optimization",
            "Sensitivity Analysis",
        ],
    ),
    (
        "Support",
        ["AI Assistant", "Security and Trusted Files", "Troubleshooting", "About"],
    ),
]


class HelpWidget(QtWidgets.QWidget):
    """Two-panel help browser with title and content search."""

    @staticmethod
    def _sidebar_style(theme: str) -> str:
        colors = THEMES[str(theme).lower()]
        selected_text = "#ffffff"
        return f"""
            QWidget#helpSidebar {{ background:{colors["bg_panel"]}; }}
            QTreeWidget {{
                background:{colors["bg_panel"]}; color:{colors["text_main"]};
                border:none; outline:0; font-size:12px;
            }}
            QTreeWidget::item {{ padding:4px 6px; color:{colors["text_main"]}; }}
            QTreeWidget::item:selected {{
                background:{colors["primary"]}; color:{selected_text};
            }}
            QTreeWidget::item:hover:!selected {{
                background:{colors["bg_input"]}; color:{colors["text_main"]};
            }}
            QLineEdit {{
                background:{colors["bg_input"]}; border:1px solid {colors["border"]};
                border-radius:5px; color:{colors["text_main"]}; padding:6px 8px;
            }}
            QLineEdit:focus {{ border-color:{colors["primary"]}; }}
        """

    def __init__(self) -> None:
        super().__init__()
        self._current_key = "Start Here"
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = QtWidgets.QWidget()
        self._sidebar.setObjectName("helpSidebar")
        self._sidebar.setFixedWidth(230)
        side_layout = QtWidgets.QVBoxLayout(self._sidebar)
        side_layout.setContentsMargins(9, 11, 9, 9)
        side_layout.setSpacing(7)
        self._title = QtWidgets.QLabel("PyLCSS Help")
        side_layout.addWidget(self._title)

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
        self._divider = QtWidgets.QFrame()
        self._divider.setFrameShape(QtWidgets.QFrame.VLine)
        self._divider.setFixedWidth(1)
        root.addWidget(self._sidebar)
        root.addWidget(self._divider)
        root.addWidget(self._browser, 1)

        self._build_tree()
        first = self._tree.topLevelItem(0).child(0)
        self._tree.setCurrentItem(first)
        self.apply_theme(current_theme())

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
                visible = (
                    not query or query in (key + " " + _PAGES.get(key, "")).casefold()
                )
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
        self._current_key = key
        html = _PAGES.get(key, _page(f"<h1>{key}</h1><p>Page not found.</p>"))
        theme = getattr(self, "_active_theme", current_theme())
        self._browser.setHtml(self._themed_html(html, theme))
        self._browser.verticalScrollBar().setValue(0)

    @staticmethod
    def _themed_html(html, theme):
        if str(theme).lower() != "light":
            return html
        colors = THEMES["light"]
        light_css = f"""
body {{ background:#ffffff; color:{colors["text_main"]}; }}
h1 {{ color:{colors["primary"]}; border-bottom-color:{colors["border"]}; }}
h2 {{ color:{colors["primary_hover"]}; }}
b {{ color:{colors["text_main"]}; }}
code,pre {{
    background:{colors["bg_dark"]}; border-color:{colors["border"]};
    color:{colors["text_main"]};
}}
th {{ background:#e8edf3; color:{colors["text_main"]}; }}
th,td {{ border-color:{colors["border"]}; }}
tr:nth-child(even) td {{ background:#f7f8fa; }}
a {{ color:{colors["primary"]}; }}
.tip {{
    background:#edf3f8; border-left-color:{colors["primary"]};
    color:{colors["text_main"]};
}}
.warn {{
    background:#fff7e8; border-left-color:{colors["danger"]};
    color:{colors["text_main"]};
}}
"""
        return html.replace("</style>", light_css + "</style>", 1)

    def apply_theme(self, theme):
        """Refresh HTML colors and navigation accents."""
        theme = str(theme).lower()
        self._active_theme = theme
        colors = THEMES[theme]
        self._sidebar.setStyleSheet(self._sidebar_style(theme))
        self._title.setStyleSheet(
            f"color:{colors['primary']}; font-size:15px; font-weight:700;"
            "padding:2px 4px 5px;"
        )
        self._browser.setStyleSheet(
            f"background:{colors['bg_panel']}; color:{colors['text_main']}; border:none;"
        )
        self._divider.setStyleSheet(f"color:{colors['border']};")
        section_color = colors["primary"] if theme == "light" else "#82b8ef"
        for index in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(index).setForeground(0, QtGui.QColor(section_color))
        self._show_page(self._current_key)
