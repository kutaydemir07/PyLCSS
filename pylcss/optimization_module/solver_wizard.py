# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Solver selection wizard and helper utilities for optimization.

This module provides an interactive wizard to help users select the most
appropriate optimization solver based on their problem characteristics.
"""

from PySide6 import QtWidgets, QtCore, QtGui
import qtawesome as qta
from pylcss.config import SOLVER_DESCRIPTIONS


class SolverSelectionWizard(QtWidgets.QDialog):
    """
    Interactive wizard to guide users in selecting the best optimization solver.
    
    Asks questions about problem characteristics and recommends suitable solvers
    based on the responses.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Solver Selection Wizard")
        self.setMinimumSize(700, 600)
        self.setModal(True)
        
        self.selected_solver = None
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("Optimization Solver Selection Wizard")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QtWidgets.QLabel(
            "Answer the following questions to find the best solver for your problem"
        )
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Question 1: Problem size
        grp_size = QtWidgets.QGroupBox("1. How many design variables does your problem have?")
        layout_size = QtWidgets.QVBoxLayout(grp_size)
        
        self.radio_small = QtWidgets.QRadioButton("Small (1-20 variables)")
        self.radio_medium = QtWidgets.QRadioButton("Medium (20-100 variables)")
        self.radio_large = QtWidgets.QRadioButton("Large (100+ variables)")
        self.radio_small.setChecked(True)
        
        layout_size.addWidget(self.radio_small)
        layout_size.addWidget(self.radio_medium)
        layout_size.addWidget(self.radio_large)
        layout.addWidget(grp_size)
        
        # Question 2: Constraints
        grp_constraints = QtWidgets.QGroupBox("2. Does your problem have constraints?")
        layout_constraints = QtWidgets.QVBoxLayout(grp_constraints)
        
        self.radio_no_constraints = QtWidgets.QRadioButton("No constraints (or bounds only)")
        self.radio_yes_constraints = QtWidgets.QRadioButton("Yes, general constraints (inequalities/equalities)")
        self.radio_no_constraints.setChecked(True)
        
        layout_constraints.addWidget(self.radio_no_constraints)
        layout_constraints.addWidget(self.radio_yes_constraints)
        layout.addWidget(grp_constraints)
        
        # Question 3: Function smoothness
        grp_smooth = QtWidgets.QGroupBox("3. Is your objective function smooth and differentiable?")
        layout_smooth = QtWidgets.QVBoxLayout(grp_smooth)
        
        self.radio_smooth = QtWidgets.QRadioButton("Yes (smooth, continuous)")
        self.radio_noisy = QtWidgets.QRadioButton("No (noisy, discontinuous, or has 'if' statements)")
        self.radio_unknown = QtWidgets.QRadioButton("Uncertain / Don't know")
        self.radio_smooth.setChecked(True)
        
        layout_smooth.addWidget(self.radio_smooth)
        layout_smooth.addWidget(self.radio_noisy)
        layout_smooth.addWidget(self.radio_unknown)
        layout.addWidget(grp_smooth)
        
        # Question 4: Landscape
        grp_landscape = QtWidgets.QGroupBox("4. Does your problem have multiple local optima?")
        layout_landscape = QtWidgets.QVBoxLayout(grp_landscape)
        
        self.radio_single_optimum = QtWidgets.QRadioButton("No (single optimum / convex)")
        self.radio_multi_modal = QtWidgets.QRadioButton("Yes (multiple peaks/valleys)")
        self.radio_landscape_unknown = QtWidgets.QRadioButton("Unknown")
        self.radio_single_optimum.setChecked(True)
        
        layout_landscape.addWidget(self.radio_single_optimum)
        layout_landscape.addWidget(self.radio_multi_modal)
        layout_landscape.addWidget(self.radio_landscape_unknown)
        layout.addWidget(grp_landscape)
        
        # Question 5: Time budget
        grp_time = QtWidgets.QGroupBox("5. What is your time budget for optimization?")
        layout_time = QtWidgets.QVBoxLayout(grp_time)
        
        self.radio_fast = QtWidgets.QRadioButton("Fast (< 1 minute)")
        self.radio_moderate = QtWidgets.QRadioButton("Moderate (1-10 minutes)")
        self.radio_slow = QtWidgets.QRadioButton("Slow (10+ minutes, high accuracy)")
        self.radio_moderate.setChecked(True)
        
        layout_time.addWidget(self.radio_fast)
        layout_time.addWidget(self.radio_moderate)
        layout_time.addWidget(self.radio_slow)
        layout.addWidget(grp_time)
        
        layout.addSpacing(10)
        
        # Recommendation area
        self.recommendation_label = QtWidgets.QLabel()
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet("""
            QLabel {
                background-color: #e8f4f8;
                color: #1a1a1a;
                border: 2px solid #4a90e2;
                border-radius: 5px;
                padding: 15px;
            }
        """)
        layout.addWidget(self.recommendation_label)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.btn_recommend = QtWidgets.QPushButton(qta.icon('fa5s.magic'), " Get Recommendation")
        self.btn_recommend.clicked.connect(self.show_recommendation)
        btn_layout.addWidget(self.btn_recommend)
        
        self.btn_select = QtWidgets.QPushButton(qta.icon('fa5s.check'), " Select Solver")
        self.btn_select.clicked.connect(self.accept)
        self.btn_select.setEnabled(False)
        btn_layout.addWidget(self.btn_select)
        
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(btn_layout)
    
    def show_recommendation(self):
        """Analyze responses and recommend a solver."""
        # Determine problem characteristics
        size = self._get_problem_size()
        has_constraints = self.radio_yes_constraints.isChecked()
        is_smooth = self.radio_smooth.isChecked()
        is_noisy = self.radio_noisy.isChecked()
        is_multi_modal = self.radio_multi_modal.isChecked()
        time_budget = self._get_time_budget()
        
        # Decision logic
        recommended_solvers = []
        
        if is_multi_modal or self.radio_landscape_unknown.isChecked():
            # Multi-modal or unknown landscape -> global optimizers
            if time_budget == 'slow':
                recommended_solvers.append(('Differential Evolution', 10))
            recommended_solvers.append(('Nevergrad', 9))
            if not has_constraints and is_smooth:
                recommended_solvers.append(('Nelder-Mead', 6))
        
        elif is_noisy or self.radio_unknown.isChecked():
            # Noisy or unknown smoothness -> derivative-free
            if has_constraints:
                recommended_solvers.append(('COBYLA', 9))
                recommended_solvers.append(('Nevergrad', 8))
            else:
                recommended_solvers.append(('Nelder-Mead', 8))
                recommended_solvers.append(('Powell', 7))
                recommended_solvers.append(('COBYLA', 6))
        
        elif is_smooth:
            # Smooth function -> gradient-based
            if has_constraints:
                if size == 'small' or size == 'medium':
                    recommended_solvers.append(('SLSQP', 10))
                    recommended_solvers.append(('trust-constr', 8))
                else:
                    recommended_solvers.append(('trust-constr', 9))
                    recommended_solvers.append(('SLSQP', 7))
            else:
                if size == 'large':
                    recommended_solvers.append(('L-BFGS-B', 10))
                    recommended_solvers.append(('TNC', 8))
                else:
                    recommended_solvers.append(('L-BFGS-B', 9))
                    recommended_solvers.append(('SLSQP', 8))
                    recommended_solvers.append(('TNC', 7))
        
        # Sort by score and get top recommendations
        recommended_solvers.sort(key=lambda x: x[1], reverse=True)
        
        if not recommended_solvers:
            recommended_solvers = [('SLSQP', 5)]  # Default fallback
        
        # Display recommendation
        top_solver = recommended_solvers[0][0]
        self.selected_solver = top_solver
        
        solver_info = SOLVER_DESCRIPTIONS.get(top_solver, {})
        
        recommendation_text = f"""<h3>🎯 Recommended Solver: {top_solver}</h3>
<p><b>{solver_info.get('name', 'N/A')}</b></p>
<p><i>{solver_info.get('description', 'N/A')}</i></p>

<p><b>Why this solver?</b><br>
{solver_info.get('when_to_use', 'Best match for your problem characteristics.')}</p>

<p><b>Best for:</b> {solver_info.get('best_for', 'General optimization')}<br>
<b>Speed:</b> {solver_info.get('speed', 'N/A')}<br>
<b>Robustness:</b> {solver_info.get('robustness', 'N/A')}</p>
"""
        
        if len(recommended_solvers) > 1:
            alternatives = ", ".join([s[0] for s in recommended_solvers[1:3]])
            recommendation_text += f"<p><b>Alternatives:</b> {alternatives}</p>"
        
        self.recommendation_label.setText(recommendation_text)
        self.btn_select.setEnabled(True)
    
    def _get_problem_size(self):
        if self.radio_small.isChecked():
            return 'small'
        elif self.radio_medium.isChecked():
            return 'medium'
        else:
            return 'large'
    
    def _get_time_budget(self):
        if self.radio_fast.isChecked():
            return 'fast'
        elif self.radio_moderate.isChecked():
            return 'moderate'
        else:
            return 'slow'
    
    def get_selected_solver(self):
        """Return the recommended solver name."""
        return self.selected_solver


class ConvergenceDiagnostics:
    """
    Utility class for analyzing optimization convergence.
    
    Provides methods to detect stagnation, oscillation, and other convergence
    issues during optimization runs.
    """
    
    @staticmethod
    def analyze_convergence(iteration_history, cost_history, patience=50, rel_tol=1e-4):
        """
        Analyze convergence behavior from optimization history.
        
        Args:
            iteration_history: List of iteration numbers
            cost_history: List of corresponding objective values
            patience: Number of iterations without improvement to trigger warning
            rel_tol: Relative tolerance for improvement detection
        
        Returns:
            dict: Diagnostic information with warnings and suggestions
        """
        diagnostics = {
            'is_converged': False,
            'is_stagnant': False,
            'is_oscillating': False,
            'warnings': [],
            'suggestions': []
        }
        
        if len(cost_history) < 10:
            return diagnostics
        
        # Check for stagnation
        recent_costs = cost_history[-patience:]
        if len(recent_costs) >= patience:
            best_recent = min(recent_costs)
            best_overall = min(cost_history)
            
            if abs(best_recent - best_overall) / (abs(best_overall) + 1e-10) > rel_tol:
                # Recent values not improving
                diagnostics['is_stagnant'] = True
                diagnostics['warnings'].append(
                    f"⚠️ Stagnation detected: No significant improvement in last {patience} iterations"
                )
                diagnostics['suggestions'].append(
                    "• Try tightening tolerances or increasing max iterations"
                )
                diagnostics['suggestions'].append(
                    "• Consider switching to a global optimizer (Nevergrad, Differential Evolution)"
                )
        
        # Check for oscillation
        if len(cost_history) >= 20:
            recent = cost_history[-20:]
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
            
            if sign_changes > len(diffs) * 0.7:  # More than 70% oscillation
                diagnostics['is_oscillating'] = True
                diagnostics['warnings'].append(
                    "⚠️ Oscillation detected: Objective value fluctuating significantly"
                )
                diagnostics['suggestions'].append(
                    "• Reduce step size or relax tolerances"
                )
                diagnostics['suggestions'].append(
                    "• Try a different solver (e.g., COBYLA for noisy problems)"
                )
        
        # Check for convergence
        if len(cost_history) >= 10:
            recent_window = cost_history[-10:]
            cost_range = max(recent_window) - min(recent_window)
            avg_cost = sum(recent_window) / len(recent_window)
            
            if cost_range / (abs(avg_cost) + 1e-10) < rel_tol:
                diagnostics['is_converged'] = True
        
        return diagnostics
