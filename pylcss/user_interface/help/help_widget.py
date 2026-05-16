# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
# Help widget for PyLCSS.

Provides comprehensive documentation and guidance for all application features
organized in tabbed sections for easy navigation.
"""

from PySide6 import QtWidgets, QtCore
import qtawesome as qta

class HelpWidget(QtWidgets.QWidget):
    """
    Help widget containing documentation for all application features.

    Organized as a tabbed interface with sections for each main application
    component, providing detailed usage instructions and feature descriptions.
    """

    def __init__(self) -> None:
        """Initialize the help widget with all documentation tabs."""
        super().__init__()

        # specialized layout for help widget
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QtWidgets.QLabel("PyLCSS Documentation")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(title)

        # Tab widget for different help sections
        self.help_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.help_tabs)

        # Add help tabs for each main section
        self._add_modeling_help()
        self._add_cad_help()
        self._add_surrogate_help()
        self._add_solution_space_help()
        self._add_optimization_help()
        self._add_sensitivity_help()
        self._add_voice_assistant_help()
        self._add_about_tab()

    def _create_browser(self, html_content: str) -> QtWidgets.QTextBrowser:
        """Helper to create a configured QTextBrowser."""
        browser = QtWidgets.QTextBrowser()
        browser.setHtml(html_content)
        
        # [CRITICAL FIX]: Allow links to open in the system default browser
        # If False (default), clicking a link tries to load it inside the widget,
        # which fails and makes the content disappear.
        browser.setOpenExternalLinks(True)
        
        # Ensure it looks good in both light/dark themes by respecting palette
        browser.setAutoFillBackground(True)
        return browser

    def _add_modeling_help(self) -> None:
        """Add comprehensive help for the Modeling Environment tab."""
        help_text = """
        <h2>Modeling Environment</h2>

        <h3>Overview</h3>
        <p>The Modeling Environment is the core of PyLCSS, offering a visual, node-based interface for constructing complex engineering systems. It combines the ease of drag-and-drop design with the power of Python scripting.</p>

        <h3>Key Capabilities</h3>
        <ul>
        <li><b>Visual Design:</b> Build systems by connecting nodes representing inputs, outputs, and operations.</li>
        <li><b>Python Integration:</b> Write custom logic using standard Python syntax with NumPy support.</li>
        <li><b>Unit Intelligence:</b> Automatic unit conversion and compatibility checking via the Pint library.</li>
        <li><b>High Performance:</b> Models are compiled into optimized vectorized code for fast execution.</li>
        </ul>

        <h3>Building a Model</h3>
        <ol>
        <li><b>Add Nodes:</b> Right-click the canvas to add Input, Output, Intermediate, or Custom Block nodes.</li>
        <li><b>Connect:</b> Drag connections between ports to define data flow.</li>
        <li><b>Configure:</b> Double-click nodes to set parameters, units, and bounds.</li>
        <li><b>Validate:</b> Use the validation tools to ensure connectivity and unit consistency.</li>
        <li><b>Build:</b> Compile the model to prepare it for analysis and optimization.</li>
        </ol>

        <h3>Node Types</h3>
        <ul>
        <li><b>Input Node:</b> Defines design variables with min/max bounds and units.</li>
        <li><b>Output Node:</b> Marks system results for analysis and optimization objectives.</li>
        <li><b>Custom Block:</b> Executes user-defined Python code. Supports complex logic and math.</li>
        <li><b>Intermediate Node:</b> Provides intermediate values with units to the system.</li>
        </ul>

        <h3>Best Practices</h3>
        <ul>
        <li><b>Define Units:</b> Always specify units for inputs and outputs to prevent physical errors.</li>
        <li><b>Modularize:</b> Break down complex systems into smaller, manageable subsystems.</li>
        <li><b>Test Blocks:</b> Verify custom code blocks individually before integrating them.</li>
        </ul>
        """
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.project-diagram'), "Modeling")

    def _add_cad_help(self) -> None:
        """Add comprehensive help for the Design Studio tab."""
        help_text = """
        <h2>Design Studio: CAD Modeling and FEM Simulation</h2>

        <h3>Overview</h3>
        <p>PyLCSS has two complementary parametric CAD authoring paths, both feeding the same node graph and downstream FEA / crash / topology optimisation:</p>
        <ul>
        <li><b>Code Part Node (CadQuery)</b> — write one short Python snippet that builds your geometry; named parameters (L, W, H, hole_d, …) are exposed as live optimisable inputs.</li>
        <li><b>FreeCAD Part Node (interactive)</b> — double-click to launch the real FreeCAD GUI as a subprocess; sketch in PartDesign, define Spreadsheet aliases, save. PyLCSS auto-imports the geometry via BREP + sidecar JSON and exposes your aliases as live parametric properties. The optimiser drives them back into FreeCAD headlessly between iterations.</li>
        </ul>

        <h3>Key Capabilities</h3>
        <ul>
        <li><b>Two authoring modes:</b> Pick CadQuery (code-first, scriptable) or FreeCAD (interactive, GUI-driven). Both produce the same downstream <code>shape</code> output.</li>
        <li><b>Node-Based Workflow:</b> Connect parts into assemblies, mesh, define materials, constraints, loads, and solve — all in one graph.</li>
        <li><b>FEM Simulation:</b> Netgen meshing + CalculiX static solver.</li>
        <li><b>Crash Simulation:</b> OpenRadioss explicit dynamics with animation playback.</li>
        <li><b>Topology Optimization:</b> SIMP with MMA solver, shape recovery, STL export.</li>
        <li><b>Export Options:</b> STEP, STL, OBJ.</li>
        </ul>

        <h3>FreeCAD Part Node Workflow</h3>
        <ol>
        <li><b>Drag in:</b> From the library panel, Geometry → "FreeCAD Part (interactive)".</li>
        <li><b>Double-click the node</b> → FreeCAD launches in a separate window on a node-owned <code>.FCStd</code> file (auto-created on first open).</li>
        <li><b>(Optional) Add Spreadsheet:</b> Spreadsheet workbench → new Spreadsheet → fill cells like <code>A1=Length, B1=50</code>, right-click the value cell → Properties → Alias = <code>L</code>. Repeat for every parameter you want PyLCSS to drive (W, hole_d, …).</li>
        <li><b>Author your part:</b> PartDesign workbench → Body → Sketch → Pad. In each value field, right-click → Expression editor → <code>Spreadsheet.L</code> (or whichever alias). A small <i>fx</i> icon confirms the binding.</li>
        <li><b>Ctrl+S</b> inside FreeCAD. The PyLCSS Mod observer (installed under <code>%APPDATA%/FreeCAD/v1-1/Mod/PyLCSS/</code>) writes a sibling <code>.brep</code> + <code>.fcmeta.json</code> the moment you save.</li>
        <li><b>PyLCSS reacts automatically:</b> a file-system watcher fires, the graph re-executes, the 3D viewer refreshes, and the spreadsheet aliases populate the node's <code>param_&lt;i&gt;_name</code> / <code>param_&lt;i&gt;_value</code> properties.</li>
        <li><b>Optimisation / sensitivity</b> can now mutate those parameters; the next graph execute headlessly pushes the new values back into FreeCAD, recomputes, saves, and re-reads the BREP. No GUI clicks in the loop.</li>
        </ol>

        <h3>FreeCAD Requirements</h3>
        <p>The FreeCAD Part node needs FreeCAD 1.x installed. From a terminal:</p>
        <pre>python scripts/install_solvers.py --only freecad</pre>
        <p>The script downloads the official Windows installer wizard from the FreeCAD GitHub release and auto-detects the install path. If you already have FreeCAD installed, the script skips the wizard and just registers the path. PyLCSS opens cleanly without FreeCAD installed — only the FreeCAD Part node is disabled.</p>

        <h3>CadQuery Code Part Node</h3>
        <ul>
        <li><b>Code:</b> any CadQuery expression assigned to <code>result</code>. Example:
            <code>result = cq.Workplane('XY').box(L, W, H).faces('&gt;Z').workplane().hole(d)</code></li>
        <li><b>Parameters:</b> <code>name=value</code> lines (one per line) become identifiers usable in the code AND live properties on the node.</li>
        <li><b>Best for:</b> reproducible scripted geometry, sharing parametric models as plain text.</li>
        </ul>

        <h3>Other Node Types</h3>
        <ul>
        <li><b>Import:</b> STEP, STL — bring in external CAD.</li>
        <li><b>Select Face:</b> text selectors (Direction, Index, Box, NearestToPoint) OR interactive click-to-pick in the viewer.</li>
        <li><b>Assembly:</b> combine multiple shape outputs into one assembly.</li>
        <li><b>Analysis:</b> Mass Properties, Bounding Box, Surface Area, Measure Distance.</li>
        <li><b>Booleans:</b> handled inside the Code Part node (cq.cut / .union / .intersect) or via Assembly + FreeCAD multi-body.</li>
        </ul>

        <h3>FEM Simulation Workflow</h3>
        <ol>
        <li><b>Create Geometry:</b> Build your 3D model using CAD nodes.</li>
        <li><b>Generate Mesh:</b> Use the Mesh node to create finite elements with Netgen.</li>
        <li><b>Define Material:</b> Set material properties (Young's modulus, Poisson's ratio, density).</li>
        <li><b>Apply Constraints:</b> Fix supports on specific faces.</li>
        <li><b>Apply Loads:</b> Define forces or pressure loads on faces.</li>
        <li><b>Solve:</b> Run the FEA solver to compute stress and displacement.</li>
        <li><b>Visualize:</b> View results with color-coded stress/displacement maps.</li>
        </ol>

        <h3>Topology Optimization</h3>
        <p>The topology optimization module uses the SIMP (Solid Isotropic Material with Penalization) method with MMA (Method of Moving Asymptotes) optimizer to find optimal material layouts. Features include:</p>
        <ul>
        <li><b>Density Filtering:</b> Smooth density fields for manufacturable designs.</li>
        <li><b>Volume Constraints:</b> Control material usage.</li>
        <li><b>Shape Recovery:</b> Extract clean geometry using marching cubes.</li>
        </ul>

        <h3>Units System</h3>
        <p>PyLCSS CAD uses a consistent unit system throughout:</p>
        <table border="1" cellpadding="5" cellspacing="0">
        <tr><th>Quantity</th><th>Unit</th><th>Examples</th></tr>
        <tr><td>Length</td><td><b>mm</b> (millimeters)</td><td>Box width: 20 = 20 mm</td></tr>
        <tr><td>Force</td><td><b>N</b> (Newtons)</td><td>Load: 1000 = 1000 N</td></tr>
        <tr><td>Stress/Pressure</td><td><b>MPa</b> (N/mm²)</td><td>Yield: 250 = 250 MPa</td></tr>
        <tr><td>Young's Modulus</td><td><b>MPa</b></td><td>Steel: 210000 = 210 GPa</td></tr>
        <tr><td>Density</td><td><b>tonne/mm³</b></td><td>Steel: 7.85e-9</td></tr>
        </table>
        <p><i>Note: This unit system (mm-N-MPa-tonne) is consistent for FEA and ensures numerical stability.</i></p>

        <h3>Best Practices</h3>
        <ul>
        <li><b>Mesh Quality:</b> Use appropriate element sizes - smaller for detailed features.</li>
        <li><b>Convergence:</b> Check that FEA results converge with mesh refinement.</li>
        <li><b>Units:</b> All dimensions are in mm. Material E in MPa, forces in N.</li>
        <li><b>Export:</b> Use STEP format for maximum compatibility with other CAD software.</li>
        </ul>
        """
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.cube'), "CAD and FEM")

    def _add_surrogate_help(self) -> None:
        """Add comprehensive help for the Surrogate Training tab."""
        help_text = """
        <h2>Surrogate Modeling</h2>

        <h3>Overview</h3>
        <p>Accelerate your simulations by replacing computationally expensive components with fast, accurate machine learning models. The Surrogate Modeling module supports a wide range of algorithms, from traditional regression to deep learning.</p>

        <h3>Supported Algorithms</h3>
        <ul>
        <li><b>PyTorch Neural Networks:</b> Deep learning models with customizable architecture and GPU support.</li>
        <li><b>Random Forest:</b> Robust ensemble method, great for non-linear relationships.</li>
        <li><b>Gradient Boosting:</b> High-performance tree-based method.</li>
        <li><b>Gaussian Process:</b> Probabilistic models providing uncertainty estimates.</li>
        <li><b>MLP Regressor:</b> Standard neural networks for simpler problems.</li>
        </ul>

        <h3>Training Workflow</h3>
        <ol>
        <li><b>Select Node:</b> Choose the system component to approximate.</li>
        <li><b>Generate Data:</b> Create a training dataset using Monte Carlo sampling.</li>
        <li><b>Configure Model:</b> Select an algorithm and tune hyperparameters (layers, learning rate, etc.).</li>
        <li><b>Train:</b> Execute the training process with real-time loss monitoring.</li>
        <li><b>Evaluate:</b> Check R² scores, RMSE, and parity plots to verify accuracy.</li>
        <li><b>Deploy:</b> Attach the trained surrogate to the node to speed up system analysis.</li>
        </ol>

        <h3>Sanity Check</h3>
        <p>Use the "Sanity Check" mode to debug your training pipeline. Try overfitting a small number of samples (1 or 10) to ensure the model architecture is capable of learning the data patterns.</p>
        """
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.brain'), "Surrogate AI")

    def _add_solution_space_help(self) -> None:
        """Add comprehensive help for the Solution Space tab."""
        help_text = """
        <h2>Solution Space Exploration</h2>

        <h3>Overview</h3>
        <p>Understand your design space through extensive Monte Carlo sampling. This module helps you visualize trade-offs, identify feasible regions, and explore the robustness of your design.</p>

        <h3>Features</h3>
        <ul>
        <li><b>Monte Carlo Sampling:</b> Rapidly evaluate thousands of design points.</li>
        <li><b>Feasibility Analysis:</b> Automatically classify designs as "Feasible" or "Infeasible" based on constraints.</li>
        <li><b>Interactive Visualization:</b> Explore data with 2D scatter plots, histograms, and parallel coordinates.</li>
        <li><b>Product Families:</b> Analyze multiple system variants simultaneously to find common platforms.</li>
        </ul>

        <h3>Analysis Steps</h3>
        <ol>
        <li><b>Configure Sampling:</b> Set the number of samples and input distributions.</li>
        <li><b>Run Simulation:</b> Execute the sampling process.</li>
        <li><b>Visualize:</b> Use the plotting tools to find correlations and design limits.</li>
        <li><b>Filter:</b> Isolate specific regions of interest for detailed analysis.</li>
        <li><b>Export:</b> Save results to CSV for external reporting.</li>
        </ol>

        <h3>Scientific Reference</h3>
        <p>The solution space computation methods implemented in this module are based on:</p>
        <p><b>Markus Zimmermann, Johannes Edler von Hoessle</b><br>
        <i>"Computing solution spaces for robust design"</i><br>
        International Journal for Numerical Methods in Engineering (2013)<br>
        DOI: <a href="https://doi.org/10.1002/nme.4450">10.1002/nme.4450</a></p>
        """
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.chart-area'), "Solution Space")

    def _add_optimization_help(self) -> None:
        """Add comprehensive help for the Optimization tab."""
        help_text = """
        <h2>Optimization</h2>

        <h3>Overview</h3>
        <p>Find the optimal design parameters that minimize or maximize your objectives while satisfying all constraints. PyLCSS provides a suite of industry-standard algorithms for both local and global optimization.</p>

        <h3>Algorithms</h3>
        <ul>
        <li><b>SLSQP:</b> Efficient gradient-based method for smooth, constrained problems.</li>
        <li><b>Nevergrad:</b> Powerful gradient-free optimizer for noisy or black-box functions.</li>
        <li><b>Differential Evolution:</b> Global optimization method robust against local minima.</li>
        <li><b>COBYLA:</b> Linear approximation method for constrained problems without derivatives.</li>
        </ul>

        <h3>Setting Up an Optimization</h3>
        <ol>
        <li><b>Objectives:</b> Select outputs to minimize or maximize. You can weight multiple objectives.</li>
        <li><b>Constraints:</b> Define limits on outputs (e.g., Stress < 200 MPa).</li>
        <li><b>Algorithm:</b> Choose the solver that best fits your problem characteristics.</li>
        <li><b>Run:</b> Execute the optimization and monitor convergence in real-time.</li>
        </ol>

        <h3>Advanced Features</h3>
        <ul>
        <li><b>Smart Scaling:</b> Automatically normalizes variables and constraints for better numerical stability.</li>
        <li><b>Multi-Start:</b> Run optimization from multiple initial points to find global optima.</li>
        <li><b>Pareto Analysis:</b> Explore trade-offs between conflicting objectives.</li>
        </ul>
        """
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.rocket'), "Optimization")

    def _add_sensitivity_help(self) -> None:
        """Add comprehensive help for the Sensitivity Analysis tab."""
        help_text = """
        <h2>Sensitivity Analysis</h2>

        <h3>Overview</h3>
        <p>Identify which input parameters have the most impact on your system's performance. Sensitivity analysis is crucial for model simplification, robust design, and prioritizing design efforts.</p>

        <h3>Methodology: Sobol Indices</h3>
        <p>PyLCSS uses variance-based Sobol sensitivity analysis, providing:</p>
        <ul>
        <li><b>First-Order Indices:</b> The direct contribution of each parameter to output variance.</li>
        <li><b>Total-Order Indices:</b> The total contribution, including interactions with other parameters.</li>
        </ul>

        <h3>Workflow</h3>
        <ol>
        <li><b>Select Outputs:</b> Choose the target results to analyze.</li>
        <li><b>Configure Samples:</b> Set the sample size (powers of 2 recommended).</li>
        <li><b>Analyze:</b> Run the Sobol analysis.</li>
        <li><b>Interpret:</b> View bar charts to identify critical parameters and interactions.</li>
        </ol>

        <h3>Why Use It?</h3>
        <ul>
        <li><b>Focus Effort:</b> Spend time optimizing only the most influential variables.</li>
        <li><b>Simplify Models:</b> Fix non-influential variables to constant values.</li>
        <li><b>Understand Physics:</b> Gain insights into the driving forces of your system.</li>
        </ul>
        """
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.chart-bar'), "Sensitivity")

    def _add_voice_assistant_help(self) -> None:
        """Add help for the AI assistant and speech input panel."""
        help_text = """
        <h2>AI Assistant</h2>

        <h3>Overview</h3>
        <p>PyLCSS ships a PydanticAI-based agent that uses <b>native LLM function-calling</b> to
        drive 25 in-app tools — CAD authoring, system-graph building, simulations,
        optimisation, sensitivity, surrogate training, and UI navigation. Strict JSON-schema
        validation on every tool call (with auto-retry on validation errors) makes even small
        local models reliable for production use.</p>

        <h3>Setup</h3>
        <ol>
        <li><b>Open Assistant:</b> Click the robot button in the top-right corner.</li>
        <li><b>Text Input:</b> Type a request and press Enter or Send.</li>
        <li><b>Voice Input:</b> Toggle Voice to stream microphone audio through the new RealtimeSTT pipeline (Silero VAD + Faster-Whisper). Partial transcripts appear live; the full utterance is dispatched as a natural-language request when you finish speaking.</li>
        <li><b>First Voice Run:</b> Whisper weights (~500 MB) download once into the user cache.</li>
        </ol>

        <h3>LLM Providers</h3>
        <p>Configure from the assistant panel's settings (gear icon). Every provider uses native function-calling — no JSON-plan parsing fallback.</p>
        <ul>
        <li><b>OpenAI:</b> GPT-4.x / GPT-5 family.</li>
        <li><b>Anthropic:</b> Claude Haiku, Sonnet, Opus 4.x.</li>
        <li><b>Google:</b> Gemini 2.5 Pro / Flash.</li>
        <li><b>Local (recommended for privacy):</b> Any OpenAI-compatible server — <b>LM Studio</b>, <b>Ollama</b>, <b>vLLM</b>. Set the base URL to e.g. <code>http://localhost:1234/v1</code> and pick a model that supports tool calling (Qwen 2.5 7B+, Llama 3.1 8B+, Mistral Nemo, GPT-OSS 20B). All voice + LLM work runs offline.</li>
        </ul>

        <h3>Voice Stack</h3>
        <ul>
        <li><b>STT:</b> RealtimeSTT wrapping Silero VAD (production-grade ONNX, ~1 ms/chunk) + Faster-Whisper. Auto-detects CUDA / MPS / CPU and picks the right compute_type (float16 / int8). Streaming partial transcripts; engineering jargon (von Mises, CalculiX, CadQuery, …) is pre-seeded into Whisper's <code>initial_prompt</code> so domain terms transcribe correctly.</li>
        <li><b>TTS:</b> RealtimeTTS with the local Kokoro-82M engine (~550× realtime on CPU, fully offline). Barge-in supported — start speaking and the assistant stops mid-sentence.</li>
        </ul>

        <h3>What the assistant can do</h3>
        <ul>
        <li>Create CadQuery code-part geometry: <i>"Make a 100x50x10 bracket with a 10 mm hole on the top face."</i></li>
        <li>Insert a FreeCAD Part node for interactive sketching: <i>"Open a new FreeCAD part for a fork bracket."</i></li>
        <li>Build a system model with inputs / outputs / custom Python blocks.</li>
        <li>Run sensitivity analysis and summarise the most influential variables.</li>
        <li>Train a surrogate (MLP / GP / Random Forest / PyTorch DNN) on the active design.</li>
        <li>Switch tabs, save the project, kick off an NSGA-II optimisation.</li>
        </ul>

        <h3>Tips</h3>
        <ul>
        <li><b>Decompose:</b> The assistant is instructed to issue parallel tool calls when steps are independent (e.g. "create geometry AND switch to FEA tab AND save"). Phrasing your request as one sentence still works.</li>
        <li><b>Local models:</b> The strict-schema retry loop means an 8B model can handle multi-step requests that used to require GPT-4. Try Qwen 2.5 7B first.</li>
        <li><b>Privacy:</b> API keys are encrypted with a machine-specific cipher in <code>llm_memory.json</code>. Pick the Local provider to keep every byte on your machine.</li>
        </ul>

        <h3>Current Limitations</h3>
        <ul>
        <li><b>File Selection:</b> Native file dialogs still require direct user selection.</li>
        <li><b>Provider Setup:</b> Cloud LLM providers require valid API credentials.</li>
        <li><b>Verification:</b> Treat generated graphs and simulation setup as drafts until validated.</li>
        </ul>
        """       
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.robot'), "AI Assistant")

    def _add_about_tab(self) -> None:
        """Add the About information as a help tab."""
        about_text = """
        <h2>About PyLCSS</h2>

        <h3>Engineering Design Optimization Platform</h3>
        <p><b>Version 2.0.0</b></p>
        <p>PyLCSS is a professional engineering platform for system modeling, parametric CAD, 
        FEA simulation, multi-disciplinary optimization, sensitivity analysis, solution-space 
        exploration, and surrogate modeling.</p>

        <h3>Core Technologies</h3>
        <ul>
        <li><b>Python & PySide6:</b> Modern, responsive desktop UI with DPI awareness.</li>
        <li><b>CadQuery & OpenCASCADE:</b> Parametric solid modeling and B-Rep CAD kernel.</li>
        <li><b>scikit-fem & Netgen:</b> Finite Element Analysis and mesh generation.</li>
        <li><b>NumPy, SciPy & scikit-learn:</b> High-performance computing and ML surrogates.</li>
        <li><b>PyTorch:</b> Deep learning surrogate models (ConfigurableNet).</li>
        <li><b>Nevergrad, NSGA-II & Multi-Start:</b> 7 optimization solvers (local, global, multi-objective).</li>
        <li><b>SALib:</b> 4 sensitivity methods (Sobol, Morris, FAST, Delta/DMIM).</li>
        <li><b>VTK & pyqtgraph:</b> 3D visualization and interactive 2D plotting.</li>
        <li><b>pint:</b> Physical unit conversion across SI, Imperial, CGS systems.</li>
        <li><b>meshio & h5py:</b> Multi-format mesh and HDF5 data I/O.</li>
        </ul>

        <h3>Key Capabilities</h3>
        <ul>
        <li>Node-based visual system modeling with 50+ CAD, math, and simulation nodes</li>
        <li>Topology optimization (SIMP with MMA/OC, symmetry, Heaviside projection)</li>
        <li>Cross-validation, hyperparameter optimization, and feature importance analysis</li>
        <li>Each tab provides its own integrated import/export capabilities</li>
        </ul>

        <h3>License & Copyright</h3>
        <p>Licensed under the PolyForm Shield License 1.0.0.</p>
        <p>Copyright © 2026 Kutay Demir. All rights reserved.</p>

        <p><i>Developed for advanced engineering research and industrial applications.</i></p>
        """
        browser = self._create_browser(about_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.info-circle'), "About")
