# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Searchable in-app guide for the workflows exposed by the current UI."""

from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.user_interface.common.theme_manager import THEMES, current_theme


_HTML_HEAD = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body { background:#1e1f22; color:#dce2eb; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       font-size:13px; line-height:1.65; margin:0; padding:20px 28px; }
.doc-container { max-width:860px; margin:0 auto; }
.crumb { color:#82b8ef; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
h1 { color:#d9a441; font-size:22px; font-weight:700; border-bottom:1px solid #343943;
     padding-bottom:10px; margin:0 0 16px; }
h2 { color:#e7bd63; font-size:16px; font-weight:600; margin:24px 0 8px; border-bottom:1px solid #282c33; padding-bottom:4px; }
h3 { color:#dce2eb; font-size:14px; font-weight:600; margin:16px 0 6px; }
p { margin:6px 0 12px; }
ul,ol { margin:6px 0 14px 22px; padding:0; }
li { margin:4px 0; }
b { color:#ffffff; font-weight:600; }
code,pre { background:#15171b; border:1px solid #343943; border-radius:4px;
           font-family:'Consolas','Courier New',monospace; font-size:12px; }
code { padding:2px 6px; color:#e6edf3; }
pre { padding:10px 14px; white-space:pre-wrap; line-height:1.45; }
table { border-collapse:collapse; width:100%; margin:12px 0 16px; }
th { background:#282c33; color:#e7bd63; text-align:left; font-weight:600; }
th,td { border:1px solid #343943; padding:8px 10px; vertical-align:top; }
tr:nth-child(even) td { background:#1a1c20; }
a { color:#65aaf2; text-decoration:none; }
a:hover { text-decoration:underline; }
kbd { background:#282c33; border:1px solid #484f58; border-radius:3px;
      box-shadow:0 1px 0 rgba(0,0,0,0.3); font-family:Consolas,monospace;
      font-size:11px; padding:2px 5px; color:#e6edf3; }
.note { background:#16243b; border-left:3px solid #388bfd; padding:10px 14px;
        border-radius:0 4px 4px 0; margin:14px 0; color:#dce2eb; }
.tip { background:#122822; border-left:3px solid #3fb950; padding:10px 14px;
       border-radius:0 4px 4px 0; margin:14px 0; color:#dce2eb; }
.caution { background:#2e2210; border-left:3px solid #d29922; padding:10px 14px;
           border-radius:0 4px 4px 0; margin:14px 0; color:#dce2eb; }
.warn { background:#32191c; border-left:3px solid #f85149; padding:10px 14px;
        border-radius:0 4px 4px 0; margin:14px 0; color:#dce2eb; }
.badge { display:inline-block; font-size:10px; font-weight:600; padding:2px 7px;
         border-radius:10px; margin-right:4px; }
</style></head><body>"""


def _page(body: str, category: str = "") -> str:
    crumb = f'<div class="crumb">{category}</div>' if category else ""
    return _HTML_HEAD + '<div class="doc-container">' + crumb + body + "</div></body></html>"


_PAGES: dict[str, str] = {
    "Start Here": _page("""
<h1>Start Here</h1>
<p>Open <b>PyLCSS</b> to select a workspace. The chooser window shows the workspaces available in your installation. Close the workspace window to return to the chooser.</p>
<table>
<tr><th>Workspace</th><th>What You Can Do</th></tr>
<tr><td><b>PyLCSS MDAO</b></td><td>System modeling, surrogate training, solution-space exploration, optimization, and sensitivity analysis.</td></tr>
<tr><td><b>PyLCSS CAE</b></td><td>Parametric CAD, geometry selection, meshing, static FEA, modal analysis, impact simulation, topology optimization, and lattice generation.</td></tr>
</table>
<h2>Start Work in MDAO</h2>
<ol>
<li>Open the <b>Modeling</b> tab.</li>
<li>Add <b>Design Variable</b>, <b>Function Block</b>, and <b>QoI</b> nodes to the canvas.</li>
<li>Connect the node ports and click <b>Validate</b>.</li>
<li>Click <b>Build Model</b> to compile the system definition.</li>
<li>Open the target analysis tab to start your investigation.</li>
</ol>
<h2>Start Work in CAE</h2>
<ol>
<li>Click <b>New</b> or <b>Open</b> on the workbench toolbar to load a CAD study.</li>
<li>Add geometry, mesh, material, boundary conditions, and solver nodes.</li>
<li>Select the terminal solver node.</li>
<li>Press <kbd>F5</kbd> or click <b>Run</b>.</li>
<li>Inspect results in the 3D viewport and save the study before export.</li>
</ol>
<div class="note"><b>Navigation note:</b> MDAO pages open from the top navigation row. CAE opens directly into the Design Studio workbench. Use the MDAO <b>Help</b> button or CAE <b>Help &rarr; User Guide</b> to open this guide.</div>
""", "Getting Started"),

    "Installation and Workspaces": _page("""
<h1>Installation and Workspaces</h1>
<p>Run the Windows setup installer and choose one option: <b>PyLCSS MDAO only</b>, <b>PyLCSS CAE only</b>, or <b>PyLCSS MDAO and CAE</b>. Setup installs a shared Python runtime and required packages.</p>
<p>CalculiX, OpenCourant, and FreeCAD are optional external tools. Install only the external tools your workflows require.</p>
<h2>Install External Solvers</h2>
<p>For a source installation in an activated environment, run the solver installer script:</p>
<pre>python scripts/install_solvers.py --only ccx
python scripts/install_solvers.py --only opencourant
python scripts/install_solvers.py --only freecad</pre>
<h2>Launch Options</h2>
<p>The main PyLCSS shortcut opens the workspace chooser. Dedicated shortcuts open MDAO or CAE directly. From a command prompt:</p>
<pre>PyLCSS.exe --workspace mdao
PyLCSS.exe --workspace cae</pre>
<h2>Source Installation</h2>
<p>From the repository root, install the desired workspace extras:</p>
<pre>pip install -e ".[mdao]"
pip install -e ".[cae]"
pip install -e ".[mdao,cae]"</pre>
<p>Start PyLCSS with <code>python -m pylcss.main</code>, <code>python -m pylcss.mdao</code>, or <code>python -m pylcss.cae</code>.</p>
<div class="caution"><b>Dependency note:</b> A Function Block in MDAO that evaluates a saved CAE study requires both workspaces and the external solver configured for that study.</div>
""", "Getting Started"),

    "Projects and Files": _page("""
<h1>Projects and Files</h1>
<h2>MDAO Project Folder</h2>
<p>In MDAO, select <b>File &rarr; Save Project</b>. This creates a folder named after the product and saves Modeling, Surrogates, Solution Space, Optimization, and Sensitivity state. Select <b>File &rarr; Load Project</b> to restore the project folder.</p>
<h2>CAE Study Files</h2>
<p>The CAE workbench stores the node graph and settings in a <code>.cad</code> file. Computed FEA, modal, impact, and topology results are stored beside it in a companion <code>.cad.results.h5</code> file. Always keep the study file and its companion HDF5 file together when moving projects.</p>
<h2>Combined Workflows</h2>
<p>Load the project folder in MDAO. To open linked CAD models in CAE, select <b>Open</b> on the CAE toolbar and choose <code>design_studio.cad</code> from that project folder.</p>
<h2>Computed Results Export</h2>
<p>Click <b>Save Results</b> in CAE to export computed results to JSON or HDF5. To export geometry, use the STEP, IGES, or STL actions.</p>
""", "Getting Started"),

    "Interface": _page("""
<h1>Interface and Layout</h1>
<h2>MDAO Navigation</h2>
<p>Use the top navigation row to switch pages: <b>Modeling</b>, <b>Surrogates</b>, <b>Solution Space</b>, <b>Optimization</b>, <b>Sensitivity</b>, and <b>Help</b>. Click a name to open its page. An underline marks the active page.</p>
<h2>CAE Workbench Layout</h2>
<ul>
<li><b>Page Navigation:</b> Top navigation row to switch between <b>Setup</b> (CAD modeling, node graph, and physics solvers) and <b>Results</b> (comprehensive data tables, convergence curves, and mechanical KPIs).</li>
<li><b>Left Panel:</b> Dual-tab dock featuring the hierarchical <b>Study Tree</b> outliner and searchable <b>Library</b> catalog.</li>
<li><b>Center Area:</b> Flexible workspace with <b>3D Focus</b>, <b>Split View</b> (Side-by-Side or Stacked), and <b>Graph Focus</b> layout modes.</li>
<li><b>Right Panel:</b> Collapsible Inspector dock with node <b>Properties</b> and simulation <b>Results</b>.</li>
</ul>
<h2>Keyboard Shortcuts</h2>
<table>
<tr><th>Action</th><th>Shortcut</th></tr>
<tr><td>Run selected study or node</td><td><kbd>F5</kbd></td></tr>
<tr><td>Stop running computation</td><td><kbd>Shift</kbd> + <kbd>F5</kbd></td></tr>
<tr><td>Fit graph / Reset 3D view</td><td><kbd>F</kbd> / <kbd>R</kbd></td></tr>
<tr><td>Undo / Redo</td><td><kbd>Ctrl</kbd> + <kbd>Z</kbd> / <kbd>Ctrl</kbd> + <kbd>Y</kbd></td></tr>
<tr><td>Delete selected nodes</td><td><kbd>Delete</kbd></td></tr>
</table>
<p>In MDAO, choose <b>View &rarr; Theme</b>. In CAE, use the <b>Theme</b> menu. Both workspaces support Dark and Light themes.</p>
""", "Getting Started"),

    "Design Studio Overview": _page("""
<h1>Design Studio Overview</h1>
<p>Design Studio is the parametric workbench inside PyLCSS CAE. It uses a node graph for parametric CAD, physics simulation, post-processing, and manufacturing export.</p>
<h2>Geometry Sources</h2>
<ul>
<li><b>Parametric Solids:</b> Box, Cylinder, Tube, Torus, shell, Boolean, hole, fillet, and pattern nodes.</li>
<li><b>Code Part:</b> CadQuery Python scripts with exposed scalar parameters.</li>
<li><b>FreeCAD Part:</b> Node-owned FreeCAD parametric documents (when FreeCAD is installed).</li>
<li><b>Import:</b> STEP, IGES, BREP, or surface mesh files.</li>
</ul>
<h2>Execution Behavior</h2>
<p>With one node selected, click <b>Run</b> or press <kbd>F5</kbd> to evaluate that node and its upstream inputs. With no selection, the active solver workflow runs. Sibling solver branches never run implicitly.</p>
<div class="tip"><b>Fast preview:</b> Property changes update a CAD-only preview. Heavy FEA, modal, impact, and topology calculations start only when you click Run.</div>
<h2>Available Solvers</h2>
<ul>
<li><b>Static FEA:</b> CalculiX solid solver for linear and non-linear stress analysis.</li>
<li><b>Modal Solver:</b> CalculiX natural frequency extraction and mode shape visualization.</li>
<li><b>Impact:</b> OpenCourant transient non-linear crash solver.</li>
<li><b>Topology:</b> SIMP density optimization with compact, on-demand CAD recovery.</li>
<li><b>Lattice:</b> Homogenized cellular synthesis with 3MF beam and STEP export.</li>
</ul>
""", "CAE — CAD and Simulation"),

    "Geometry Selection": _page("""
<h1>Geometry Selection</h1>
<p>Boundary conditions reference CAD surfaces and curves through <b>Select Geometry</b> or <b>Pick Geometry</b> nodes. Choose <b>Face</b>, <b>Edge</b>, or <b>Vertex</b>.</p>
<h2>Selection Procedure</h2>
<ol>
<li>Connect the upstream CAD shape to <b>Select Geometry</b>.</li>
<li>Choose the entity type. For interactive picking, click the 3D pick button.</li>
<li>Select one or more entities in the 3D viewport and accept the selection.</li>
<li>Connect the selection output to a support, load, or condition node.</li>
<li>Select any condition or solver node to inspect the persistent 3D overlay.</li>
</ol>
<h2>Connection Rules</h2>
<p>Supports and point loads accept faces, edges, or vertices. Pressure loads require an area and accept faces only. Bearing loads require cylindrical internal faces.</p>
<div class="warn"><b>Selection stability:</b> Upstream CAD modifications can renumber raw entity indices. Direction filters, nearest-point filters, bounding boxes, or interactive picks provide more stable references.</div>
""", "CAE — CAD and Simulation"),

    "Meshing": _page("""
<h1>Meshing</h1>
<p>The <b>Mesh</b> node uses Netgen through OpenCASCADE to create solid tetrahedral or shell meshes. Local sizing is supported on selected CAD faces.</p>
<table>
<tr><th>Mesh Type</th><th>Element Type</th><th>Recommended Application</th></tr>
<tr><td><b>Tet (Linear)</b></td><td>C3D4</td><td>Fast preliminary checks and coarse screening.</td></tr>
<tr><td><b>Tet10 (Quadratic)</b></td><td>C3D10</td><td>Static solid stress, bending, and modal frequency analysis.</td></tr>
<tr><td><b>Shell Triangles</b></td><td>S3 / Thin Shell</td><td>OpenCourant crash models. The CAD input must be a true midsurface.</td></tr>
</table>
<p>The node rejects collapsed elements and calculates mean-ratio quality. Always inspect element quality and perform mesh convergence studies on key design values.</p>
<div class="warn"><b>Midsurface warning:</b> Meshing the outer skin of a thick solid as a Shell does not extract a physical midsurface. Use dedicated midsurface geometry or import shell sheets.</div>
""", "CAE — CAD and Simulation"),

    "Static FEA": _page("""
<h1>Static FEA</h1>
<p>The <b>Static Solver</b> node runs CalculiX for structural stress and displacement analysis.</p>
<h2>Required Connections</h2>
<table>
<tr><th>Input Port</th><th>Connected Source</th></tr>
<tr><td><code>mesh</code></td><td>Solid tetrahedral mesh using Tet (C3D4) or Tet10 (C3D10) elements</td></tr>
<tr><td><code>material</code></td><td>Material definition with elastic modulus and Poisson's ratio</td></tr>
<tr><td><code>constraints</code></td><td>One or more Support nodes</td></tr>
<tr><td><code>loads</code></td><td>Force, Pressure, Bearing Load, or Gravity nodes</td></tr>
<tr><td><code>components</code></td><td>Optional Body outputs for assemblies</td></tr>
</table>
<h2>Supported Load Types</h2>
<ul>
<li><b>Force:</b> Concentrated force vector $(F_x, F_y, F_z)$ applied to selected entities.</li>
<li><b>Pressure:</b> Uniform normal surface traction on selected faces (Inward or Outward).</li>
<li><b>Bearing Load:</b> Compressive cosine contact distribution on cylindrical faces. It models pin contact with zero tension on the back half.</li>
<li><b>Gravity:</b> Volumetric body acceleration applied across the full model.</li>
<li><b>Load Case Weighting:</b> Set <code>case_weight</code> and <code>case_name</code> for multi-case structural studies.</li>
</ul>
<h2>Solver Settings</h2>
<p>Select <b>Linear</b>, <b>Nonlinear (Geometric)</b>, or <b>Nonlinear (Plastic)</b> in the inspector. Select <b>Deck only</b> to export the CalculiX <code>.inp</code> deck without launching the solver.</p>
""", "CAE — CAD and Simulation"),

    "Modal Analysis": _page("""
<h1>Modal Analysis</h1>
<p>The <b>Modal Solver</b> node runs CalculiX natural frequency extraction and mode shape vibration analysis.</p>
<h2>Required Connections</h2>
<table>
<tr><th>Input Port</th><th>Connected Source</th></tr>
<tr><td><code>mesh</code></td><td>Solid tetrahedral mesh (Tet10 quadratic solid recommended)</td></tr>
<tr><td><code>material</code></td><td>Material with Young's modulus, Poisson's ratio, and mass density (&rho;)</td></tr>
<tr><td><code>constraints</code></td><td>Support nodes defining fixed or constrained degrees of freedom</td></tr>
</table>
<h2>Inspector Settings</h2>
<ul>
<li><b>Number of Modes (<code>num_modes</code>):</b> The count of natural frequencies to calculate (default 6).</li>
<li><b>Active Mode (<code>active_mode</code>):</b> The mode index displayed in the 3D viewport.</li>
<li><b>Deformation Scale:</b> Visual scaling multiplier (<code>Auto</code>, <code>1x</code>, <code>5x</code>, <code>10x</code>, <code>50x</code>, <code>100x</code>).</li>
<li><b>Deck Only:</b> Generates the CalculiX <code>*FREQUENCY</code> input deck without executing the solver.</li>
</ul>
<h2>Interactive Mode Shape Table</h2>
<p>After a run, the inspector displays the <b>Natural Frequencies & Mode Shapes</b> table listing Mode number, Frequency in Hertz (Hz), Angular frequency (rad/s), and Period (s).</p>
<div class="tip"><b>Interactive preview:</b> Click any row in the table to display that mode shape eigenvector in the 3D viewport immediately.</div>
""", "CAE — CAD and Simulation"),

    "Impact": _page("""
<h1>Impact and Crashworthiness</h1>
<p>The <b>Impact Solver</b> node runs dynamic transient impact simulations with OpenCourant.</p>
<h2>Scenario Setup</h2>
<ul>
<li><b>Fixed specimen + moving impactor:</b> Requires impact face selection and rear support.</li>
<li><b>Moving body + fixed wall:</b> Specify body velocity; supports are not required.</li>
<li><b>Prescribed moving wall:</b> Requires impact face selection; rear support is not required.</li>
</ul>
<p>Velocity uses mm/ms (numerically equivalent to m/s); mass is defined on Impact Setup; End Time uses ms.</p>
<h2>Crash Performance Indicators (KPIs)</h2>
<p>The Results panel computes standard vehicle crashworthiness metrics:</p>
<ul>
<li><b>Crash Force Efficiency (CFE):</b> Ratio of mean crushing force to peak force ($F_{\\text{mean}} / F_{\\text{peak}}$). Displays qualitative status: &ge;70% Excellent (Progressive Buckling), &ge;50% Good, &lt;50% Low Efficiency.</li>
<li><b>Specific Energy Absorption (SEA):</b> Absorbed energy per unit mass in kJ/kg ($E_{\\text{abs}} / \\text{mass}$).</li>
<li><b>Peak Crushing Force ($F_{\\text{peak}}$) & Mean Force ($F_{\\text{mean}}$):</b> Displayed in kN.</li>
<li><b>Useful Crush Stroke & Absorbed Energy:</b> Total deformation distance (mm) and energy (kJ).</li>
<li><b>Eroded Elements:</b> Number of failed elements removed by constitutive criteria.</li>
</ul>
<h2>Solution Quality and Energy Gates</h2>
<p>The solver qualifies the explicit solution through automated physical checks:</p>
<ul>
<li><b>Energy Balance Error:</b> Must remain &le; 5% for a qualified solution.</li>
<li><b>Hourglass Energy Ratio:</b> Must remain &le; 10% of total energy.</li>
<li><b>Hourglass + Contact Energy:</b> Combined artificial dissipation must remain &le; 15%.</li>
<li><b>Added Mass Ratio:</b> Mass scaling added mass must remain &le; 1%.</li>
</ul>
<div class="tip"><b>Interactive curves:</b> Click <b>View Time-History Curves</b> to open the interactive chart dialog for force-displacement ($F$-$d$), energy components, and deceleration pulses.</div>
""", "CAE — CAD and Simulation"),

    "Topology Optimization": _page("""
<h1>Topology Optimization</h1>
<p>The <b>Topology Solver</b> optimizes solid material distribution directly from a CAD design domain using the SIMP method.</p>
<h2>Study Configuration</h2>
<p>Connect the design domain, material, supports, forces, passive regions, and joints. A <b>Passive Region</b> node protects critical CAD volumes as solid material or void clearance.</p>
<h2>Smooth CAD Reconstruction</h2>
<p>The recovered smooth surface appears as soon as optimization finishes and exports directly to STL. Select CAD or STEP export to prepare a CAD body separately. Geometric recognition recovers planes, cylinders, and extruded closed profiles first; an extruded profile with holes can use just two cap faces and one face per profile. General freeform shapes use checked surface recovery. Compatible surfaces are consolidated into larger selectable faces.</p>
<ol>
<li><b>Recovered Boundary:</b> Physical voxel contours with feature-constrained smoothing.</li>
<li><b>Quad Surface:</b> Instant Meshes creates field-aligned quadrilaterals.</li>
<li><b>Sparse Layouts:</b> Generalized motorcycle graphs group the surface into larger regions.</li>
<li><b>Surface Fit:</b> Harmonic parameterization, adaptive sampling and NURBS fitting with shared boundaries and multi-resolution control.</li>
<li><b>Validation:</b> Sampled geometric error, orientation, material volume, cavity preservation, and STEP reimport.</li>
</ol>
<p>CAD conversion checks material volume, cavities, sampled surface deviation, and STEP reimport. If conversion cannot preserve the shape, the recovered surface remains available for STL export. Conversion progress shows preparation stages.</p>
<p>The result views are <b>Density</b> and <b>CAD</b>. After solving TopOpt, use <b>Generate Lattice</b> in its inspector to choose a cell family and create graded cells within the optimized envelope. Preserved interfaces remain solid. <b>Lattice Infill</b> fills an existing body with uniform cells.</p>
<p>Use <b>Volume Remesh</b> to create a solid tetrahedral mesh for downstream CalculiX validation.</p>
""", "CAE — CAD and Simulation"),

    "Generate Lattice": _page("""
<h1>Generate Lattice after TopOpt</h1>
<p>Run the <b>Topology Solver</b>, then open <b>Generate Lattice</b> in its inspector. Choose the lattice type, cell size, minimum printable wall or strut thickness, boundary skin, and minimum/maximum cell density. Zero cell size selects <b>Auto fit</b>: local design thickness controls pitch, non-design pads are excluded, and features too small for printable cells stay solid. Boundary cells are trimmed and joined to the skin. Zero wall and skin dimensions use automatic sizing.</p>
<p>Generation maps the smoothed optimized density field to local cell thickness inside the retained topology envelope. It preserves solid non-design interfaces and voids. Changing a cell family and generating again uses the saved original solve; <b>Show Solid TopOpt Result</b> restores its geometry.</p>
<ul>
<li><b>Octet Truss:</b> Stretch-dominated reference for structural stiffness.</li>
<li><b>BCC:</b> Compliant strut cells.</li>
<li><b>Gyroid:</b> Continuous curved sheets.</li>
<li><b>Schwarz Primitive:</b> Sheet cells with straight channels.</li>
<li><b>Honeycomb:</b> Directional prismatic cells.</li>
</ul>
<p>No cell is best for all loads and manufacturing processes. Density mapping creates a candidate lattice; the original solid solve does not verify its stiffness, stress, buckling or fatigue. Remesh the generated geometry and validate it under the original loads. Existing solid validation is cleared on generation.</p>
<p>Export the complete generated surface as <b>STL</b>. Supported strut-only representations may also export STEP or 3MF. A skin is retained in mesh export. <b>Lattice Infill</b> remains available for uniform geometric filling of other solids.</p>
""", "CAE — CAD and Simulation"),

    "Units and Solvers": _page("""
<h1>Units and Solvers</h1>
<h2>Engineering Units</h2>
<table>
<tr><th>Quantity</th><th>Unit</th><th>Standard Convention</th></tr>
<tr><td>Length and Displacement</td><td>mm</td><td>Model geometry coordinates</td></tr>
<tr><td>Force</td><td>N</td><td>Point loads, reactions, contact force</td></tr>
<tr><td>Stress, Pressure, Modulus</td><td>MPa</td><td>1 MPa = 1 N/mm<sup>2</sup></td></tr>
<tr><td>Mass Density (&rho;)</td><td>tonne/mm<sup>3</sup></td><td>Steel is approximately 7.85 &times; 10<sup>-9</sup></td></tr>
<tr><td>Impact Time and Velocity</td><td>ms and mm/ms</td><td>1 mm/ms = 1 m/s</td></tr>
</table>
<h2>External Solvers Installation</h2>
<p>Install external tools using the setup utility script:</p>
<pre>python scripts/install_solvers.py --only ccx
python scripts/install_solvers.py --only opencourant
python scripts/install_solvers.py --only freecad</pre>
<p>PyLCSS detects solvers automatically. Check node library tooltips to inspect detection status.</p>
""", "CAE — CAD and Simulation"),

    "Modeling Environment": _page("""
<h1>Modeling Environment</h1>
<p>The <b>Modeling</b> tab builds multidisciplinary system graphs from interconnected functional blocks.</p>
<table>
<tr><th>Node Type</th><th>Role and Function</th></tr>
<tr><td><b>Design Variable</b></td><td>Input parameter with nominal value, lower bound, upper bound, and engineering unit</td></tr>
<tr><td><b>Intermediate</b></td><td>Calculated internal quantity passed between functional blocks</td></tr>
<tr><td><b>Function Block</b></td><td>Python calculation block or linked PyLCSS Design Studio CAD study</td></tr>
<tr><td><b>QoI</b></td><td>Quantity of Interest representing system performance, requirements, or constraints</td></tr>
</table>
<h2>Workflow Procedure</h2>
<ol>
<li>Create or load a system model.</li>
<li>Add nodes and connect compatible output-to-input ports.</li>
<li>Double-click nodes to configure equations or parameters.</li>
<li>Click <b>Validate</b> to check for graph cycles and type mismatches.</li>
<li>Click <b>Build Model</b> to compile and publish the system to downstream analysis tabs.</li>
</ol>
<p>A Design Studio Function Block links a saved <code>.cad</code> file, maps variables to CAD parameters, and executes the target solver terminal.</p>
""", "MDAO — System Workflows"),

    "Surrogate Training": _page("""
<h1>Surrogate Training</h1>
<p>The <b>Surrogates</b> tab replaces expensive simulation runs with fast mathematical response surfaces.</p>
<h2>Scientific Auto Mode</h2>
<p>Auto Mode benchmarks and optimizes state-of-the-art surrogate modeling architectures:</p>
<ul>
<li><b>Ordinary and Universal Kriging:</b> Gaussian process regression with anisotropic ARD.</li>
<li><b>KPLS:</b> Kriging by Partial Least Squares for high-dimensional models.</li>
<li><b>Radial Basis Functions (RBF):</b> Thin Plate and Polyharmonic splines with polynomial drift.</li>
<li><b>Sparse PCE:</b> Sparse Orthogonal Polynomial Chaos Expansion.</li>
<li><b>Support Vector Regression (SVR):</b> Anisotropic &epsilon;-insensitive SVR.</li>
<li><b>Goel-Haftka Ensemble:</b> Optimal convex combination minimizing cross-validation error covariance.</li>
</ul>
<h2>Deep Learning Architectures</h2>
<p>Design neural networks using the visual architecture editor:</p>
<ul>
<li><b>Residual Blocks:</b> Deep skip connections for stable gradient propagation.</li>
<li><b>Fourier Feature Embeddings:</b> Random Fourier features for high-frequency physics responses.</li>
<li><b>PyTorch Script Compilation:</b> Compile and train user-defined PyTorch network classes.</li>
</ul>
<h2>Data Import Wizard</h2>
<p>Click <b>Import Data</b> to load external CSV or JSON datasets. The dialog provides delimiter detection, column mapping, header validation, and data preview.</p>
""", "MDAO — System Workflows"),

    "Solution Space": _page("""
<h1>Solution Space</h1>
<p>The <b>Solution Space</b> tab calculates decoupled intervals and feasible regions where all QoI requirements are met.</p>
<h2>Workflow Steps</h2>
<ol>
<li>Select a compiled system model.</li>
<li>Review variables and QoI definitions.</li>
<li>Set lower and upper performance requirements on each QoI.</li>
<li>Select sample size and compute the solution space.</li>
<li>Inspect feasible design boxes and export results.</li>
</ol>
<h2>Multi-Dimensional Visualization</h2>
<ul>
<li><b>1-D Interval Views:</b> Visualizes feasible bounds on individual parameters.</li>
<li><b>2-D Scatter Plots:</b> Interactive scatter plots with editable box boundaries.</li>
<li><b>3-D Projected Boxes:</b> 3-variable projected bounding boxes rendered in pyqtgraph without OpenGL dependencies.</li>
<li><b>Multi-Modal Discovery:</b> Identifies disconnected feasible regions.</li>
<li><b>Product Family Design:</b> Compares shared platform components against model variants.</li>
</ul>
""", "MDAO — System Workflows"),

    "Optimization": _page("""
<h1>Optimization</h1>
<p>The <b>Optimization</b> tab solves single-objective and multi-objective constrained design problems.</p>
<h2>Supported Algorithms</h2>
<table>
<tr><th>Algorithm</th><th>Classification</th><th>Recommended Use</th></tr>
<tr><td><b>SLSQP</b></td><td>Gradient-based local</td><td>Smooth, continuous engineering problems with non-linear constraints.</td></tr>
<tr><td><b>Nevergrad</b></td><td>Stochastic global</td><td>Bounded black-box, noisy, or non-smooth response spaces.</td></tr>
<tr><td><b>Multi-Start</b></td><td>Multi-point local</td><td>Non-convex response surfaces with multiple local optima.</td></tr>
<tr><td><b>NSGA-II</b></td><td>Evolutionary Pareto</td><td>Multi-objective trade-off studies generating Pareto-optimal fronts.</td></tr>
</table>
<h2>Objective Scaling</h2>
<p>In multi-objective optimization, objectives are normalized by their initial magnitude or explicit reference scales before weights are applied. This prevents unit discrepancies from biasing the search.</p>
<div class="caution"><b>Verification note:</b> Optimization does not validate the underlying model. Always re-evaluate optimal points, active constraints, and multiple random seeds.</div>
""", "MDAO — System Workflows"),

    "Sensitivity Analysis": _page("""
<h1>Sensitivity Analysis</h1>
<p>The <b>Sensitivity</b> tab performs global sensitivity analysis using <b>Sobol variance decomposition</b> with Saltelli sequence sampling. It identifies influential design variables and quantifies parameter interactions.</p>
<h2>Estimated Indices</h2>
<table>
<tr><th>Index</th><th>Symbol</th><th>Description</th></tr>
<tr><td><b>First-Order Index</b></td><td>S<sub>1</sub></td><td>Main direct contribution of an individual variable to output variance without interactions.</td></tr>
<tr><td><b>Total-Order Index</b></td><td>S<sub>T</sub></td><td>Total contribution of a variable, including all higher-order interactions.</td></tr>
<tr><td><b>Second-Order Index</b></td><td>S<sub>2</sub></td><td>Pairwise coupling variance between pairs of variables (optional).</td></tr>
</table>
<h2>Configuration Steps</h2>
<ol>
<li>Select the target output, or select <b>All outputs</b> to run a shared sample design.</li>
<li>Select a power-of-2 base sample size (256, 1024, or 4096).</li>
<li>Enable <b>Calculate second-order interactions</b> if you require pairwise coupling data.</li>
<li>Set an optional integer random seed for reproducible sampling.</li>
<li>Click <b>Run Analysis</b>.</li>
</ol>
<h2>Visualizations and Results</h2>
<ul>
<li><b>Main & Total Effects Bar Chart:</b> Shows S<sub>1</sub> and S<sub>T</sub> indices with 95% confidence intervals.</li>
<li><b>S<sub>2</sub> Interaction Heatmap:</b> Visualizes the symmetric pairwise coupling matrix between design variables.</li>
<li><b>Ranked Sensitivity Table:</b> Ranks variables by total-order importance (S<sub>T</sub>) and highlights dominant factors.</li>
</ul>
<div class="note"><b>Sample budget note:</b> The total number of model evaluations is $N \\times (2D + 2)$ with second-order interactions, or $N \\times (D + 2)$ for first and total order only, where $D$ is the number of design variables.</div>
""", "MDAO — System Workflows"),

    "Security and Trusted Files": _page("""
<h1>Security and Trusted Files</h1>
<p>Design Studio models can include custom Python code inside <b>Code Part</b> nodes. This code executes with user system permissions during evaluation.</p>
<p>PyLCSS asks for user confirmation before executing code from an external project.</p>
<h2>Serialized Model Safety</h2>
<p>Files serialized with joblib, pickle, dill, or PyTorch can execute code during loading. Open project files, custom scripts, model weights, and solver decks only from trusted sources.</p>
<div class="warn"><b>Security warning:</b> File extensions do not form a security boundary. Always inspect Code nodes before approving execution.</div>
""", "Support"),

    "Troubleshooting": _page("""
<h1>Troubleshooting</h1>
<h2>Workspace is Unavailable</h2>
<p>Run setup again and select the missing workspace. For source installations, install the corresponding extras package (<code>.[mdao]</code> or <code>.[cae]</code>).</p>
<h2>Cannot Find CAD in MDAO</h2>
<p>Close MDAO to return to the workspace chooser, then click PyLCSS CAE. Or use the dedicated CAE desktop shortcut.</p>
<h2>Solver Does Not Start</h2>
<ul>
<li>Hover over the solver node in the library to inspect executable detection paths.</li>
<li>Verify external solver installations using <code>python scripts/install_solvers.py</code>.</li>
<li>Check <b>Deck Only</b> to verify that the input deck generates correctly.</li>
</ul>
<h2>FEA Study Fails</h2>
<ul>
<li>Verify that the solid mesh connects to the solver node.</li>
<li>Check that boundary supports eliminate all rigid body degrees of freedom.</li>
<li>Confirm that pressure loads connect to valid face selections.</li>
</ul>
<h2>3D Viewport is Blank</h2>
<p>Run the upstream CAD geometry node before opening downstream condition nodes. Interactive picking requires an evaluated geometry preview.</p>
""", "Support"),

    "About": _page("""
<h1>About PyLCSS</h1>
<p><b>PyLCSS 2.2.0.2</b> is an integrated engineering desktop software platform for parametric CAD, structural FEA, modal vibration, crash simulation, topology optimization, surrogate modeling, solution spaces, and multidisciplinary design optimization.</p>
<p>Copyright &copy; 2026 Kutay Demir.</p>
<p>Licensed under the <b>PolyForm Shield License 1.0.0</b>. See LICENSE and NOTICE files.</p>
""", "Support"),
}

_NAV = [
    (
        "Getting Started",
        ["Start Here", "Installation and Workspaces", "Projects and Files", "Interface"],
    ),
    (
        "CAE — CAD and Simulation",
        [
            "Design Studio Overview",
            "Geometry Selection",
            "Meshing",
            "Static FEA",
            "Modal Analysis",
            "Impact",
            "Topology Optimization",
            "Generate Lattice",
            "Units and Solvers",
        ],
    ),
    (
        "MDAO — System Workflows",
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
        ["Security and Trusted Files", "Troubleshooting", "About"],
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
            QTreeWidget::item {{ padding:5px 7px; color:{colors["text_main"]}; border-radius:4px; }}
            QTreeWidget::item:selected {{
                background:{colors["primary"]}; color:{selected_text};
            }}
            QTreeWidget::item:hover:!selected {{
                background:{colors["bg_input"]}; color:{colors["text_main"]};
            }}
            QLineEdit {{
                background:{colors["bg_input"]}; border:1px solid {colors["border"]};
                border-radius:5px; color:{colors["text_main"]}; padding:6px 9px; font-size:12px;
            }}
            QLineEdit:focus {{ border-color:{colors["primary"]}; }}
        """

    def __init__(self) -> None:
        super().__init__()
        self._current_key = "Start Here"
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self._splitter.setHandleWidth(1)

        self._sidebar = QtWidgets.QWidget()
        self._sidebar.setObjectName("helpSidebar")
        self._sidebar.setMinimumWidth(200)
        side_layout = QtWidgets.QVBoxLayout(self._sidebar)
        side_layout.setContentsMargins(10, 12, 10, 10)
        side_layout.setSpacing(8)
        self._title = QtWidgets.QLabel("PyLCSS Help")
        side_layout.addWidget(self._title)

        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("Search help...")
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

        self._splitter.addWidget(self._sidebar)
        self._splitter.addWidget(self._browser)
        self._splitter.setSizes([240, 810])
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)

        root.addWidget(self._splitter)

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
    def _themed_html(html: str, theme: str) -> str:
        if str(theme).lower() != "light":
            return html
        colors = THEMES["light"]
        light_css = f"""
body {{ background:#ffffff; color:{colors["text_main"]}; }}
.crumb {{ color:{colors["primary"]}; }}
h1 {{ color:{colors["primary"]}; border-bottom-color:{colors["border"]}; }}
h2 {{ color:{colors["primary_hover"]}; border-bottom-color:#e1e4e8; }}
h3 {{ color:{colors["text_main"]}; }}
b {{ color:{colors["text_main"]}; }}
code,pre {{
    background:{colors["bg_dark"]}; border-color:{colors["border"]};
    color:{colors["text_main"]};
}}
th {{ background:#e8edf3; color:{colors["text_main"]}; }}
th,td {{ border-color:{colors["border"]}; }}
tr:nth-child(even) td {{ background:#f7f8fa; }}
a {{ color:{colors["primary"]}; }}
kbd {{
    background:#f6f8fa; border-color:{colors["border"]};
    color:{colors["text_main"]}; box-shadow:0 1px 0 rgba(0,0,0,0.1);
}}
.note {{
    background:#edf3f8; border-left-color:{colors["primary"]};
    color:{colors["text_main"]};
}}
.tip {{
    background:#eaf8ed; border-left-color:#1a7f37;
    color:{colors["text_main"]};
}}
.caution {{
    background:#fff9e6; border-left-color:#b08800;
    color:{colors["text_main"]};
}}
.warn {{
    background:#fff1f0; border-left-color:{colors["danger"]};
    color:{colors["text_main"]};
}}
"""
        return html.replace("</style>", light_css + "</style>", 1)

    def apply_theme(self, theme: str) -> None:
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
        if hasattr(self, "_splitter"):
            self._splitter.setStyleSheet(
                f"QSplitter::handle {{ background: {colors['border']}; width: 1px; }}"
            )
        section_color = colors["primary"] if theme == "light" else "#82b8ef"
        for index in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(index).setForeground(0, QtGui.QColor(section_color))
        self._show_page(self._current_key)
