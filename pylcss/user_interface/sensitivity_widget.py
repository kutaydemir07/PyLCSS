# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Sensitivity Analysis Interface for PyLCSS.

Provides a GUI for performing global sensitivity analysis using Sobol indices
to identify which design variables have the most impact on system outputs.
"""

import warnings
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
import qtawesome as qta
import logging

# Suppress SALib FutureWarning about pd.unique
warnings.filterwarnings("ignore", message="unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated", category=FutureWarning, module="SALib")

from ..optimization.sensitivity import SensitivityAnalyzer
from ..optimization.evaluator import ModelEvaluator
from ..optimization.core import Variable

logger = logging.getLogger(__name__)

class SensitivityAnalysisWidget(QtWidgets.QWidget):
    """
    Widget for performing global sensitivity analysis.

    This widget allows users to:
    - Select output variables for analysis
    - Configure sample sizes
    - Run sensitivity analysis
    - Visualize results with bar charts
    """

    def __init__(self, optimization_widget=None):
        """Initialize the sensitivity analysis widget."""
        super().__init__()
        self.optimization_widget = optimization_widget
        
        # --- FIX: Safe Initialization ---
        try:
            self.analyzer = SensitivityAnalyzer()
            self.salib_available = True
        except ImportError:
            self.analyzer = None
            self.salib_available = False
            
        self.current_results = None

        self.setup_ui()
        
        # Disable UI if SALib is missing
        if not self.salib_available:
            self.lbl_status.setText("Error: SALib not installed. Sensitivity Analysis disabled.")
            self.lbl_status.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px;")
            self.btn_analyze.setEnabled(False)
            self.combo_outputs.setEnabled(False)

    def setup_ui(self):
        """Set up the user interface."""
        layout = QtWidgets.QHBoxLayout(self)

        # --- LEFT PANEL: Configuration ---
        config_panel = QtWidgets.QWidget()
        config_panel.setFixedWidth(380)
        config_layout = QtWidgets.QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(config_panel)

        # 1. Select Outputs
        grp_output = QtWidgets.QGroupBox("Select Outputs:")
        l_output = QtWidgets.QVBoxLayout(grp_output)
        self.combo_outputs = QtWidgets.QComboBox()
        self.combo_outputs.setToolTip("Select the system output variable to analyze for sensitivity. The sensitivity analysis will determine which input parameters have the most influence on this output.")
        self.btn_refresh_outputs = QtWidgets.QPushButton("Refresh Outputs")
        self.btn_refresh_outputs.setToolTip("Refresh the list of available output variables from the current optimization problem setup.")
        self.btn_refresh_outputs.clicked.connect(self.refresh_outputs)
        l_output.addWidget(self.combo_outputs)
        l_output.addWidget(self.btn_refresh_outputs)
        config_layout.addWidget(grp_output)

        # 2. Analysis Configuration
        grp_config = QtWidgets.QGroupBox("Analysis Configuration")
        l_config = QtWidgets.QFormLayout(grp_config)
        self.spin_samples = QtWidgets.QSpinBox()
        self.spin_samples.setRange(64, 8192)  # Powers of 2 from 2^6 to 2^13
        self.spin_samples.setValue(1024)  # Default to 2^10
        self.spin_samples.setSingleStep(256)  # Step by 2^8
        self.spin_samples.setToolTip("Number of samples for sensitivity analysis. Higher values provide more accurate results but take longer to compute. Must be a power of 2 for optimal Sobol sequence convergence.")
        l_config.addRow("Sample Size:", self.spin_samples)
        
        # Add note about automatic adjustment
        note_label = QtWidgets.QLabel("Note: Sample size will be automatically adjusted to the nearest power of 2 for optimal convergence.")
        note_label.setStyleSheet("font-size: 10px; color: #666; font-style: italic;")
        note_label.setWordWrap(True)
        l_config.addRow("", note_label)
        
        config_layout.addWidget(grp_config)

        # 3. Action Buttons
        self.btn_analyze = QtWidgets.QPushButton(qta.icon('fa5s.chart-bar'), " Run Sensitivity Analysis")
        self.btn_analyze.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_analyze.setToolTip("Run the sensitivity analysis using Sobol indices to quantify the influence of each input variable on the selected output.")
        self.btn_analyze.clicked.connect(self.run_analysis)
        config_layout.addWidget(self.btn_analyze)

        self.btn_export = QtWidgets.QPushButton(qta.icon('fa5s.download'), " Export Results")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("Export the sensitivity analysis results to CSV format for further analysis or reporting.")
        self.btn_export.clicked.connect(self.export_results)
        config_layout.addWidget(self.btn_export)

        config_layout.addStretch()

        # --- RIGHT PANEL: Visualization ---
        viz_panel = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_panel)

        # Status
        self.lbl_status = QtWidgets.QLabel("Status: Ready to analyze.")
        self.lbl_status.setStyleSheet("background-color: #333; color: #fff; padding: 10px; border-radius: 4px;")
        viz_layout.addWidget(self.lbl_status)

        # Canvas for Sensitivity Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMinimumSize(400, 300)
        self.plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.plot_widget.setLabel('left', 'Sensitivity Index')
        self.plot_widget.setLabel('bottom', 'Design Variables')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        viz_layout.addWidget(self.plot_widget)

        # Results Table
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Variable", "First Order", "Total Order", "Confidence"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        viz_layout.addWidget(self.results_table)

        # Progress
        self.progress = QtWidgets.QProgressBar()
        viz_layout.addWidget(self.progress)

        layout.addWidget(config_panel)
        layout.addWidget(viz_panel)

    def refresh_outputs(self):
        """Refresh the list of available output variables."""
        self.combo_outputs.clear()

        if not self.optimization_widget or not self.optimization_widget.problem:
            return

        problem = self.optimization_widget.problem

        # Add quantities of interest as potential outputs
        for qoi in problem.quantities_of_interest:
            self.combo_outputs.addItem(f"{qoi['name']} ({qoi['unit']})", qoi['name'])

        # If no QOIs, try to get from system model outputs
        if self.combo_outputs.count() == 0 and problem.system_model:
            # Disable button during refresh
            self.btn_refresh_outputs.setEnabled(False)
            self.btn_refresh_outputs.setText("Refreshing...")
            
            # Run model evaluation in background thread to prevent UI blocking
            self.refresh_worker = OutputRefreshWorker(problem)
            self.refresh_worker.done_sig.connect(self.on_outputs_refreshed)
            self.refresh_worker.error_sig.connect(self.on_refresh_error)
            self.refresh_worker.start()

    def run_analysis(self):
        """Run the sensitivity analysis."""
        if not self.salib_available:
            QtWidgets.QMessageBox.warning(self, "Error", "SALib is not installed. Sensitivity analysis is disabled.")
            return
            
        if not self.optimization_widget or not self.optimization_widget.problem:
            QtWidgets.QMessageBox.warning(self, "Error", "No optimization problem loaded.")
            return

        problem = self.optimization_widget.problem
        output_idx = self.combo_outputs.currentIndex()
        if output_idx < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No output variable selected.")
            return

        output_name = self.combo_outputs.itemData(output_idx)
        n_samples = self.spin_samples.value()

        # Adjust sample size to nearest power of 2 for optimal Sobol convergence
        original_n = n_samples
        n_samples = self._adjust_to_power_of_two(n_samples)
        
        # Update UI to reflect the adjusted sample size
        if n_samples != original_n:
            self.spin_samples.setValue(n_samples)
            self.lbl_status.setText(f"Adjusted sample size to {n_samples} (nearest power of 2) for optimal convergence.")

        self.btn_analyze.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)

        # Run analysis in a separate thread
        # Scaling disabled as per user request
        scaling = False
        self.worker = SensitivityWorker(problem, output_name, n_samples, scaling)
        self.worker.progress_sig.connect(self.update_progress)
        self.worker.done_sig.connect(self.analysis_finished)
        self.worker.start()

    def update_progress(self, value, message):
        """Update progress bar and status."""
        self.progress.setValue(value)
        self.lbl_status.setText(message)

    def analysis_finished(self, results, error):
        """Handle completion of sensitivity analysis."""
        self.btn_analyze.setEnabled(True)

        if error:
            QtWidgets.QMessageBox.critical(self, "Analysis Failed", error)
            self.lbl_status.setText("Error occurred.")
            return

        self.current_results = results
        self.btn_export.setEnabled(True)

        # Update status
        self.lbl_status.setText("Analysis complete. Results shown below.")

        # Update plot
        self.update_plot()

        # Update table
        self.update_table()

    def update_plot(self):
        """Update the sensitivity plot."""
        if not self.current_results:
            return

        self.plot_widget.clear()

        variables = self.current_results['variable_names']
        total_indices = self.current_results['total_order']
        first_indices = self.current_results['first_order']
        confidence = self.current_results.get('confidence_total', None)

        n_vars = len(variables)
        x = np.arange(n_vars)
        width = 0.35

        # Create bar graph items for first order and total order
        # First order bars (blue)
        first_bars = pg.BarGraphItem(
            x=x - width/2, 
            height=first_indices, 
            width=width, 
            brush=pg.mkBrush('#87CEEB'),  # skyblue
            pen=pg.mkPen('k', width=1)
        )
        self.plot_widget.addItem(first_bars)

        # Total order bars (coral)
        total_bars = pg.BarGraphItem(
            x=x + width/2, 
            height=total_indices, 
            width=width, 
            brush=pg.mkBrush('#F08080'),  # lightcoral
            pen=pg.mkPen('k', width=1)
        )
        self.plot_widget.addItem(total_bars)

        # Add error bars for total order if confidence intervals available
        if confidence is not None:
            error_bars = pg.ErrorBarItem(
                x=x + width/2,
                y=total_indices,
                top=confidence,
                bottom=confidence,
                beam=width*0.5,
                pen=pg.mkPen('k', width=1)
            )
            self.plot_widget.addItem(error_bars)

        # Set up x-axis labels
        ax = self.plot_widget.getAxis('bottom')
        ticks = [(i, var) for i, var in enumerate(variables)]
        ax.setTicks([ticks])
        ax.setStyle(tickTextOffset=10)
        
        # Rotate labels if there are many variables
        if n_vars > 5:
            ax.setLabel(angle=-45)

        # Add legend
        legend = self.plot_widget.addLegend(offset=(10, 10))
        legend.addItem(first_bars, 'First Order')
        legend.addItem(total_bars, 'Total Order')

        # Set title
        self.plot_widget.setTitle('Global Sensitivity Analysis')

    def update_table(self):
        """Update the results table."""
        if not self.current_results:
            return

        variables = self.current_results['variable_names']
        first_order = self.current_results['first_order']
        total_order = self.current_results['total_order']
        confidence = self.current_results.get('confidence_total', [0] * len(variables))

        self.results_table.setRowCount(len(variables))

        for i, var in enumerate(variables):
            self.results_table.setItem(i, 0, QtWidgets.QTableWidgetItem(var))
            self.results_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{first_order[i]:.4f}"))
            self.results_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{total_order[i]:.4f}"))
            self.results_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{confidence[i]:.4f}"))

        self.results_table.resizeColumnsToContents()

    def export_results(self):
        """Export sensitivity analysis results."""
        if not self.current_results:
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Sensitivity Results", "", "CSV files (*.csv);;All files (*)"
        )

        if filename:
            try:
                import pandas as pd

                data = {
                    'Variable': self.current_results['variable_names'],
                    'First_Order': self.current_results['first_order'],
                    'Total_Order': self.current_results['total_order'],
                    'Confidence': self.current_results.get('confidence_total', [0] * len(self.current_results['variable_names']))
                }

                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)

                QtWidgets.QMessageBox.information(self, "Success",
                    f"Results exported to {filename}")

            except ImportError:
                QtWidgets.QMessageBox.warning(self, "Export Error",
                    "pandas is required for CSV export. Install with: pip install pandas")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def on_outputs_refreshed(self, output_names):
        """Handle successful output refresh."""
        for output_name in output_names:
            self.combo_outputs.addItem(output_name, output_name)
        
        # Re-enable button
        self.btn_refresh_outputs.setEnabled(True)
        self.btn_refresh_outputs.setText("Refresh Outputs")

    def on_refresh_error(self, error_msg):
        """Handle error during output refresh."""
        logger.error("Error refreshing outputs: %s", error_msg)
        
        # Re-enable button
        self.btn_refresh_outputs.setEnabled(True)
        self.btn_refresh_outputs.setText("Refresh Outputs")

    def _adjust_to_power_of_two(self, n: int) -> int:
        """
        Adjust n to the nearest power of 2 for optimal Sobol sequence convergence.

        Args:
            n: Original sample count

        Returns:
            Nearest power of 2
        """
        if n <= 0:
            return 2

        # Find the nearest power of 2
        import math
        power = math.floor(math.log2(n))
        lower = 2 ** power
        upper = 2 ** (power + 1)

        # Return the closer power of 2
        if abs(n - lower) <= abs(n - upper):
            return lower
        else:
            return upper

    def save_to_folder(self, folder_path):
        """Save sensitivity settings to a folder."""
        import json
        import os
        
        json_path = os.path.join(folder_path, 'sensitivity.json')
        
        data = {
            'output_name': self.combo_outputs.currentText(),
            'n_samples': self.spin_samples.value()
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_folder(self, folder_path):
        """Load sensitivity settings from a folder."""
        import json
        import os
        
        json_path = os.path.join(folder_path, 'sensitivity.json')
        if not os.path.exists(json_path):
            return
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            self.combo_outputs.setCurrentText(data.get('output_name', ''))
            self.spin_samples.setValue(data.get('n_samples', 1024))
            
        except Exception as e:
            logger.exception("Failed to load sensitivity settings")


class OutputRefreshWorker(QtCore.QThread):
    """Worker thread for refreshing output variables list."""
    
    done_sig = QtCore.Signal(list)
    error_sig = QtCore.Signal(str)
    
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
    
    def run(self):
        try:
            # Get a sample to determine outputs
            sample_inputs = {}
            for dv in self.problem.design_variables:
                sample_inputs[dv['name']] = (dv['min'] + dv['max']) / 2
            for p in self.problem.parameters:
                sample_inputs[p['name']] = p['value']

            sample_output = self.problem.system_model(**sample_inputs)
            output_names = list(sample_output.keys())
            self.done_sig.emit(output_names)
        except Exception as e:
            self.error_sig.emit(str(e))


class SensitivityWorker(QtCore.QThread):
    """Worker thread for sensitivity analysis."""

    progress_sig = QtCore.Signal(int, str)
    done_sig = QtCore.Signal(object, str)

    def __init__(self, problem, output_name, n_samples, scaling=True):
        super().__init__()
        self.problem = problem
        self.output_name = output_name
        self.n_samples = n_samples
        self.scaling = scaling
        
        # Create ModelEvaluator for consistent evaluation and caching
        variables = [Variable(name=dv['name'], min_val=float(dv['min']), max_val=float(dv['max'])) for dv in self.problem.design_variables]
        parameters = {p['name']: p['value'] for p in self.problem.parameters}
        
        # We use scaling=False because SALib generates physical samples
        self.evaluator = ModelEvaluator(
            self.problem.system_model,
            variables,
            [], [], # No objectives/constraints
            parameters=parameters,
            scaling=False
        )
        
        # Safe initialization of analyzer
        try:
            self.analyzer = SensitivityAnalyzer()
        except ImportError:
            self.analyzer = None

    def run(self):
        try:
            if self.analyzer is None:
                raise ImportError("SALib is not installed")
                
            self.progress_sig.emit(0, "Setting up sensitivity analysis...")

            # Create problem definition for SALib
            problem_def = {
                'names': [dv['name'] for dv in self.problem.design_variables],
                'bounds': [[dv['min'], dv['max']] for dv in self.problem.design_variables]
            }

            self.progress_sig.emit(10, "Generating samples...")

            # Generate samples
            X = self.analyzer.generate_samples(problem_def, self.n_samples)

            self.progress_sig.emit(30, "Evaluating system model...")

            # Evaluate system model
            Y = []
            n_total = len(X)

            for i, x_sample in enumerate(X):
                # Create input array from sample
                x_phys = np.array(x_sample)
                
                try:
                    # Use evaluator
                    _, result, _ = self.evaluator.evaluate(x_phys)
                    output_val = result.get(self.output_name, 0.0)
                    Y.append(float(output_val))
                except Exception as e:
                    logger.warning("Error evaluating sample %d", i, exc_info=True)
                    Y.append(0.0)

                if i % 50 == 0:
                    progress = 30 + int(50 * (i / n_total))
                    self.progress_sig.emit(progress, f"Evaluating samples... ({i}/{n_total})")

            Y = np.array(Y)

            self.progress_sig.emit(80, "Analyzing sensitivity...")

            # Perform sensitivity analysis
            results = self.analyzer.analyze_sensitivity(X, Y, problem_def)

            self.progress_sig.emit(100, "Analysis complete.")
            self.done_sig.emit(results, None)

        except Exception as e:
            self.done_sig.emit(None, str(e))
