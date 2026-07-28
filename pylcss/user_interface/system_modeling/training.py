# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Background surrogate training launched from system-modeling nodes."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from pylcss.system_modeling.compiler import GraphBuilder

__all__ = ["NodeTrainingWorker", "train_selected_node_surrogate"]


class NodeTrainingWorker(QtCore.QThread):
    """
    Worker thread for training surrogate models for selected nodes.
    Generates training data by sampling the system up to the target node,
    then trains a neural network surrogate model.
    """

    progress_updated = QtCore.Signal(int, str)  # progress_percent, status_message
    training_finished = QtCore.Signal(bool, str)  # success, message

    def __init__(
        self,
        graph_builder,
        nodes,
        input_nodes,
        output_nodes,
        target_node,
        num_samples=1000,
        test_size=0.2,
        random_state=42,
    ):
        super().__init__()
        self.graph_builder = graph_builder
        self.nodes = nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.target_node = target_node
        self.num_samples = num_samples
        self.test_size = test_size
        self.random_state = random_state

    def _raise_if_cancelled(self):
        if self.isInterruptionRequested():
            raise InterruptedError("Training cancelled.")

    def run(self):
        try:
            # Import sklearn modules here to avoid numpy compatibility issues at module load
            import joblib
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error

        except ImportError:
            self.training_finished.emit(
                False,
                "Scikit-learn not installed. Please run: pip install scikit-learn joblib",
            )
            return

        try:
            self._raise_if_cancelled()
            self.progress_updated.emit(0, "Generating spy model code...")

            # Build spy model to capture training data
            spy_code, spy_inputs, spy_outputs = self.graph_builder.build_spy_model(
                self.nodes,
                self.input_nodes,
                self.output_nodes,
                self.target_node.id,
                "spy_model",
            )

            self.progress_updated.emit(10, "Compiling spy model...")

            # --- FIX START ---
            # Use a single dictionary for both globals and locals so functions can see each other
            exec_context = {"np": np}
            exec(spy_code, exec_context, exec_context)
            spy_func = exec_context["spy_model"]
            # --- FIX END ---

            self._raise_if_cancelled()
            self.progress_updated.emit(
                20, f"Generating {self.num_samples} training samples..."
            )

            # Generate training data by sampling input space
            X_data = []
            y_data = []

            # Get input bounds
            input_bounds = []
            for inp_node in self.input_nodes:
                if inp_node.has_property("input_props"):
                    props = inp_node.get_property("input_props")
                    min_val = float(props.get("min", "0.0"))
                    max_val = float(props.get("max", "10.0"))
                else:
                    min_val = float(inp_node.get_property("min"))
                    max_val = float(inp_node.get_property("max"))
                input_bounds.append((min_val, max_val))

            # Sample input space
            np.random.seed(self.random_state)
            for i in range(self.num_samples):
                self._raise_if_cancelled()
                # Generate random input sample within bounds
                sample_inputs = []
                for min_val, max_val in input_bounds:
                    sample_inputs.append(np.random.uniform(min_val, max_val))

                # Execute spy model to get corresponding outputs
                inputs_dict, outputs_dict = spy_func(*sample_inputs)

                # Extract input and output values
                X_sample = [inputs_dict[f"input_{j}"] for j in range(len(spy_inputs))]
                y_sample = [
                    outputs_dict[f"output_{j}"] for j in range(len(spy_outputs))
                ]

                X_data.append(X_sample)
                y_data.append(y_sample)

                if (i + 1) % 100 == 0:
                    progress = 20 + int(50 * (i + 1) / self.num_samples)
                    self.progress_updated.emit(
                        progress, f"Generated {i + 1}/{self.num_samples} samples..."
                    )

            X = np.array(X_data)
            y = np.array(y_data)

            self._raise_if_cancelled()
            self.progress_updated.emit(70, "Training neural network surrogate...")

            # Create and train surrogate model
            # Use pipeline with StandardScaler and MLPRegressor
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        MLPRegressor(
                            hidden_layer_sizes=(100, 50),
                            activation="relu",
                            solver="adam",
                            alpha=0.001,
                            batch_size="auto",
                            learning_rate="adaptive",
                            learning_rate_init=0.01,
                            max_iter=1000,
                            random_state=self.random_state,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=10,
                        ),
                    ),
                ]
            )

            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            self._raise_if_cancelled()
            # Train model
            model.fit(X_train, y_train)

            self._raise_if_cancelled()
            self.progress_updated.emit(90, "Evaluating model performance...")

            # Evaluate model performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            self._raise_if_cancelled()
            self.progress_updated.emit(95, "Saving surrogate model...")

            # Save model to file
            model_path = f"surrogate_{self.target_node.id.replace('-', '_')}.joblib"
            joblib.dump(model, model_path)

            # Update node properties
            self.target_node.set_property("surrogate_model_path", model_path)
            self.target_node.set_property(
                "surrogate_status", f"Trained (RMSE: {rmse:.4f})"
            )

            self.progress_updated.emit(100, "Training completed successfully!")
            self.training_finished.emit(
                True, f"Surrogate model trained successfully. RMSE: {rmse:.4f}"
            )

        except InterruptedError as e:
            self.training_finished.emit(False, str(e))

        except Exception as e:
            self.training_finished.emit(False, f"Training failed: {str(e)}")


def train_selected_node_surrogate(modeling_widget, num_samples=1000):
    """
    Train a surrogate model for the currently selected node.

    Args:
        modeling_widget: The ModelingWidget instance
        num_samples: Number of training samples to generate
    """
    # Ensure a graph is loaded
    if not modeling_widget.current_graph:
        QtWidgets.QMessageBox.warning(
            modeling_widget,
            "No Graph",
            "No graph is currently loaded. Please load or create a graph.",
        )
        return

    # Get selected nodes
    selected_nodes = modeling_widget.current_graph.selected_nodes()
    if not selected_nodes:
        QtWidgets.QMessageBox.warning(
            modeling_widget,
            "No Selection",
            "Please select a custom block node to train a surrogate model.",
        )
        return

    if len(selected_nodes) > 1:
        QtWidgets.QMessageBox.warning(
            modeling_widget,
            "Multiple Selection",
            "Please select only one node for surrogate training.",
        )
        return

    target_node = selected_nodes[0]
    if not target_node.type_.startswith("com.pfd.custom_block"):
        QtWidgets.QMessageBox.warning(
            modeling_widget,
            "Invalid Selection",
            "Please select a custom block node for surrogate training.",
        )
        return

    # Ask user for sample count
    val, ok = QtWidgets.QInputDialog.getInt(
        modeling_widget,
        "Training Settings",
        "Number of Samples:",
        value=num_samples,
        min=100,
        max=1000000,
        step=1000,
    )
    if not ok:
        return
    num_samples = val

    # Confirm training
    reply = QtWidgets.QMessageBox.question(
        modeling_widget,
        "Train Surrogate Model",
        f"Train surrogate model for node '{target_node.name()}'?\n\n"
        f"This will generate {num_samples} training samples and may take some time.",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
    )

    if reply != QtWidgets.QMessageBox.Yes:
        return

    # Get graph data
    all_nodes = modeling_widget.current_graph.all_nodes()
    input_nodes = [n for n in all_nodes if n.type_.startswith("com.pfd.input")]
    output_nodes = [n for n in all_nodes if n.type_.startswith("com.pfd.output")]

    # Pass the current_graph to GraphBuilder
    graph_builder = GraphBuilder(modeling_widget.current_graph)

    worker = NodeTrainingWorker(
        graph_builder, all_nodes, input_nodes, output_nodes, target_node, num_samples
    )

    # Create progress dialog
    progress_dialog = QtWidgets.QProgressDialog(
        "Training surrogate model...", "Cancel", 0, 100, modeling_widget
    )
    progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.setAutoReset(True)

    # Connect signals
    worker.progress_updated.connect(
        lambda progress, msg: (
            progress_dialog.setValue(progress),
            progress_dialog.setLabelText(msg),
        )
    )

    worker.training_finished.connect(
        lambda success, msg: (
            progress_dialog.close(),
            QtWidgets.QMessageBox.information(
                modeling_widget,
                "Training Complete" if success else "Training Failed",
                msg,
            )
            if success
            else QtWidgets.QMessageBox.critical(
                modeling_widget, "Training Failed", msg
            ),
            # Refresh node display if successful
            target_node.view.update() if success else None,
        )
    )

    # Handle cancellation cooperatively to avoid killing the thread mid-operation.
    progress_dialog.canceled.connect(worker.requestInterruption)

    # Start training
    worker.start()
    progress_dialog.exec_()
