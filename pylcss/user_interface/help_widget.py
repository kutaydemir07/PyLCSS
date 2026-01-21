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
        """Add comprehensive help for the CAD tab."""
        help_text = """
        <h2>CAD Modeling and FEM Simulation</h2>

        <h3>Overview</h3>
        <p>PyLCSS includes a powerful parametric CAD module built on CadQuery, enabling you to create 3D models using a node-based interface. Combined with finite element analysis (FEA) capabilities, you can design and validate mechanical components entirely within the platform.</p>

        <h3>Key Capabilities</h3>
        <ul>
        <li><b>Parametric Design:</b> Create 3D models using nodes for primitives, operations, and transformations.</li>
        <li><b>Node-Based Workflow:</b> Connect nodes to build complex geometries visually.</li>
        <li><b>FEM Simulation:</b> Perform structural analysis using scikit-fem with Netgen meshing.</li>
        <li><b>Topology Optimization:</b> Optimize material distribution for minimum compliance.</li>
        <li><b>Export Options:</b> Export to STEP, STL, and other CAD formats.</li>
        </ul>

        <h3>CAD Node Types</h3>
        <ul>
        <li><b>Primitives:</b> Box, Sphere, Cylinder, Cone - basic 3D shapes.</li>
        <li><b>Sketch Operations:</b> Rectangle, Circle, Polygon - 2D profiles for extrusion.</li>
        <li><b>Operations:</b> Extrude, Revolve, Fillet, Chamfer - modify and combine shapes.</li>
        <li><b>Boolean Operations:</b> Union, Subtract, Intersect - combine multiple parts.</li>
        <li><b>Transformations:</b> Translate, Rotate, Mirror, Pattern - position and replicate.</li>
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
        """Add comprehensive help for the Voice Assistant system."""
        help_text = """
        <h2>Voice Assistant (Hands-Free Control)</h2>

        <h3>Overview</h3>
        <p>PyLCSS includes an offline Voice Assistant powered by <b>Faster-Whisper</b> for hands-free control.
        It offers high-accuracy, real-time speech recognition running entirely on your local machine.</p>

        <h3>Setup Requirements</h3>
        <ol>
        <li><b>Enable Voice Control:</b> Toggle "Voice Control" in the Hands-Free menu.</li>
        <li><b>First Run:</b> The AI model (Whisper Base) will be downloaded automatically (approx 140MB).</li>
        </ol>

        <h3>🤖 LLM Intelligence</h3>
        <p>PyLCSS features a powerful LLM execution engine that can build systems for you.</p>
        <ol>
        <li><b>Activation:</b> Simply say your request (e.g., <i>"Create a red box"</i>) or ask a question.</li>
        <li><b>Providers:</b> Configure your preferred AI provider (OpenAI, Claude, Gemini, GPT@RUB) in the Settings panel.</li>
        <li><b>Capabilities:</b> The AI understands the full PyLCSS node library and can construct complex graphs, answer engineering questions, and control the UI.</li>
        <li><b>Privacy:</b> Securely encrypts your API keys locally.</li>
        </ol>

        <h3>Mouse Actions</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"click", "left click"</td><td>Left mouse click</td></tr>
        <tr><td>"right click"</td><td>Right mouse click (context menu)</td></tr>
        <tr><td>"double click"</td><td>Double left click</td></tr>
        <tr><td>"drag", "drop"</td><td>Toggle drag mode</td></tr>
        </table>

        <h3>Scrolling</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"scroll up", "scroll down"</td><td>Scroll content</td></tr>
        <tr><td>"scroll left", "scroll right"</td><td>Horizontal scroll</td></tr>
        <tr><td>"page up", "page down"</td><td>Fast scroll</td></tr>
        </table>

        <h3>Tab Navigation</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"go to modeling"</td><td>Switch to Modeling Environment</td></tr>
        <tr><td>"go to cad"</td><td>Switch to CAD Environment</td></tr>
        <tr><td>"go to surrogate"</td><td>Switch to Surrogate Training</td></tr>
        <tr><td>"go to solution space"</td><td>Switch to Solution Space</td></tr>
        <tr><td>"go to optimization"</td><td>Switch to Optimization</td></tr>
        <tr><td>"go to sensitivity"</td><td>Switch to Sensitivity Analysis</td></tr>
        <tr><td>"go to help"</td><td>Switch to Help & Documentation</td></tr>
        <tr><td>"next tab", "previous tab"</td><td>Navigate between tabs</td></tr>
        </table>

        <h3>Keyboard Shortcuts</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"save", "save project"</td><td>Ctrl+S</td></tr>
        <tr><td>"load project", "open project"</td><td>Ctrl+O</td></tr>
        <tr><td>"undo", "redo"</td><td>Ctrl+Z / Ctrl+Y</td></tr>
        <tr><td>"copy", "paste", "cut"</td><td>Ctrl+C / Ctrl+V / Ctrl+X</td></tr>
        <tr><td>"delete"</td><td>Delete key</td></tr>
        <tr><td>"escape"</td><td>Cancel/close dialogs</td></tr>
        <tr><td>"enter"</td><td>Confirm/submit</td></tr>
        </table>

        <h3>Modeling Environment</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"add input", "add design variable"</td><td>Create Input node</td></tr>
        <tr><td>"add output", "add qoi"</td><td>Create Output node</td></tr>
        <tr><td>"add function"</td><td>Create Custom Block</td></tr>
        <tr><td>"add intermediate"</td><td>Create Intermediate node</td></tr>
        <tr><td>"new system", "add system"</td><td>Add new system variant</td></tr>
        <tr><td>"next system", "previous system"</td><td>Switch between systems</td></tr>
        <tr><td>"validate graph"</td><td>Check model connectivity</td></tr>
        <tr><td>"build model"</td><td>Compile model</td></tr>
        <tr><td>"clear graph"</td><td>Delete all nodes</td></tr>
        </table>

        <h3>CAD Environment</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"add box", "add cube"</td><td>Create Box primitive</td></tr>
        <tr><td>"add cylinder"</td><td>Create Cylinder</td></tr>
        <tr><td>"add sphere"</td><td>Create Sphere</td></tr>
        <tr><td>"add cone"</td><td>Create Cone</td></tr>
        <tr><td>"add torus", "add donut"</td><td>Create Torus</td></tr>
        <tr><td>"add extrude"</td><td>Add Extrude operation</td></tr>
        <tr><td>"add fillet", "add chamfer"</td><td>Edge modifications</td></tr>
        <tr><td>"add boolean", "add union", "add cut"</td><td>Boolean operations</td></tr>
        <tr><td>"run cad", "build cad"</td><td>Execute CAD graph</td></tr>
        <tr><td>"export cad", "export stl"</td><td>Export to file</td></tr>
        </table>

        <h3>Solution Space</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"generate samples"</td><td>Run Monte Carlo sampling</td></tr>
        <tr><td>"resample"</td><td>Regenerate samples</td></tr>
        <tr><td>"add plot", "clear plots"</td><td>Manage plots</td></tr>
        <tr><td>"generate dependency graph"</td><td>Compute ADG</td></tr>
        </table>

        <h3>Surrogate Training</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"train surrogate"</td><td>Start training</td></tr>
        <tr><td>"stop training"</td><td>Cancel training</td></tr>
        <tr><td>"generate data"</td><td>Create training data</td></tr>
        <tr><td>"save surrogate"</td><td>Attach trained model</td></tr>
        </table>

        <h3>Optimization & Sensitivity</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"run optimization"</td><td>Start optimization</td></tr>
        <tr><td>"stop optimization"</td><td>Cancel optimization</td></tr>
        <tr><td>"run sensitivity"</td><td>Run Sobol analysis</td></tr>
        <tr><td>"export results"</td><td>Save results to file</td></tr>
        </table>

        <h3>Control Commands</h3>
        <table border="1" cellpadding="5" cellspacing="0" width="100%">
        <tr><th>Command</th><th>Action</th></tr>
        <tr><td>"pause", "stop"</td><td>Pause voice recognition</td></tr>
        <tr><td>"resume", "start"</td><td>Resume recognition</td></tr>
        <tr><td>"start dictation"</td><td>Enable typing mode</td></tr>
        <tr><td>"stop dictation"</td><td>Return to command mode</td></tr>
        <tr><td>"minimize", "maximize"</td><td>Window controls</td></tr>
        </table>

        <h3>Tips for Best Recognition</h3>
        <ul>
        <li><b>Speak Clearly:</b> Enunciate commands distinctly</li>
        <li><b>Short Phrases:</b> Use 1-3 word commands when possible</li>
        <li><b>Wait for Beep:</b> Audio feedback confirms recognition</li>
        <li><b>Quiet Environment:</b> Minimize background noise</li>
        </ul>

        <h3>Current Limitations</h3>
        <ul>
        <li><b>No Node Connections:</b> Connecting nodes in graphs requires mouse interaction</li>
        <li><b>No Direct File Selection:</b> "Load project" opens file dialog requiring mouse</li>
        <li><b>No System Switching by Name:</b> Use "next system" / "previous system" to navigate</li>
        </ul>

        <p><i>Note: Fuzzy matching is enabled - slight variations in pronunciation are accepted.</i></p>
        """       
        browser = self._create_browser(help_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.microphone'), "Voice Assistant")

    def _add_about_tab(self) -> None:
        """Add the About information as a help tab."""
        about_text = """
        <h2>About PyLCSS</h2>

        <h3> Engineering Platform</h3>
        <p><b>Version 1.3.0</b></p>
        <p>PyLCSS is a cutting-edge tool for system modeling, simulation, and optimization. It bridges the gap between visual design and advanced computational analysis.</p>

        <h3>Core Technologies</h3>
        <ul>
        <li><b>Python & PySide6:</b> Modern, responsive user interface.</li>
        <li><b>NumPy & SciPy:</b> High-performance numerical computing.</li>
        <li><b>PyTorch:</b> State-of-the-art deep learning capabilities.</li>
        <li><b>Nevergrad:</b> Advanced gradient-free optimization.</li>
        <li><b>SALib:</b> Global sensitivity analysis.</li>
        </ul>

        <h3>License & Copyright</h3>
        <p>Licensed under the PolyForm Shield License 1.0.0.</p>
        <p>Copyright © 2026 Kutay Demir. All rights reserved.</p>

        <p><i>Developed for advanced engineering research and industrial applications.</i></p>
        """
        browser = self._create_browser(about_text)
        self.help_tabs.addTab(browser, qta.icon('fa5s.info-circle'), "About")