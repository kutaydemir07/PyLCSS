# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Sensitivity Analysis Module for PyLCSS.

This module provides global sensitivity analysis using Sobol indices
to identify which design variables have the most impact on system outputs.
"""

import warnings
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pyqtgraph as pg

# Suppress SALib FutureWarning about pd.unique
warnings.filterwarnings("ignore", message="unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated", category=FutureWarning, module="SALib")

# Try to import SALib for sensitivity analysis
try:
    from SALib.sample import sobol, morris as morris_sample
    from SALib.analyze import sobol as sobol_analyze, morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

class SensitivityAnalyzer:
    """
    Performs global sensitivity analysis using Morris screening and Sobol indices.

    This class implements a two-stage workflow:
    1. Morris Method (Screening): Quickly identify negligible variables.
    2. Sobol Method (Variance-based): Detailed analysis on important variables.
    """

    def __init__(self):
        if not SALIB_AVAILABLE:
            raise ImportError("SALib is required for sensitivity analysis. Install with: pip install SALib")

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

    def run_screening(self, problem_definition: Dict[str, Any], n_trajectories: int = 20) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run Morris screening method to identify important variables.
        
        Args:
            problem_definition: Problem definition dictionary
            n_trajectories: Number of trajectories for Morris method (default 20)
            
        Returns:
            Tuple of (important_variable_names, morris_results)
        """
        # Convert problem definition to SALib format
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }
        
        # Generate Morris samples
        X = morris_sample.sample(problem, n_trajectories, num_levels=4)
        
        return X, problem

    def analyze_screening(self, X: np.ndarray, Y: np.ndarray, problem: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Analyze Morris screening results.
        
        Args:
            X: Input samples
            Y: Output values
            problem: SALib problem definition
            
        Returns:
            Tuple of (important_variable_names, results)
        """
        si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)
        
        # Determine important variables based on mu_star
        # Threshold: variables with mu_star > 1% of max(mu_star) are considered important
        mu_star = np.array(si['mu_star'])
        max_mu = np.max(mu_star)
        threshold = 0.01 * max_mu
        
        important_indices = np.where(mu_star > threshold)[0]
        important_vars = [problem['names'][i] for i in important_indices]
        
        # If no variables meet threshold (e.g. flat output), return all
        if not important_vars:
            important_vars = problem['names']
            
        return important_vars, si

    def generate_samples(self, problem_definition: Dict[str, Any], n_samples: int = 1000) -> np.ndarray:
        """
        Generate samples for sensitivity analysis using Sobol sequences.

        Args:
            problem_definition: Dictionary defining the problem bounds
            n_samples: Number of samples to generate (should be a power of 2 for optimal convergence)

        Returns:
            Array of samples for sensitivity analysis
        """
        # Convert problem definition to SALib format
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }

        # Generate Sobol samples
        X = sobol.sample(problem, n_samples)
        return X

    def analyze_sensitivity(self, X: np.ndarray, Y: np.ndarray,
                          problem_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Sobol sensitivity analysis.

        Args:
            X: Input samples (n_samples, n_vars)
            Y: Output values (n_samples,)
            problem_definition: Problem definition dictionary

        Returns:
            Dictionary containing sensitivity indices
        """
        # Convert to SALib problem format
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }

        # Perform Sobol analysis
        Si = sobol_analyze.analyze(problem, Y, print_to_console=False)

        # Extract results
        results = {
            'variable_names': problem_definition['names'],
            'first_order': Si['S1'],
            'total_order': Si['ST'],
            'second_order': Si['S2'],
            'confidence_first': Si['S1_conf'],
            'confidence_total': Si['ST_conf']
        }

        return results

    def plot_sensitivity_indices(self, results: Dict[str, Any],
                               output_name: str = "Output") -> Dict[str, Any]:
        """
        Create a bar chart of sensitivity indices.

        Args:
            results: Results from analyze_sensitivity
            output_name: Name of the output variable

        Returns:
            Dictionary containing plot data for PyQtGraph rendering
        """
        variables = results['variable_names']
        total_indices = results['total_order']
        first_indices = results['first_order']
        confidence = results.get('confidence_total', None)

        # Return plot data for PyQtGraph rendering
        plot_data = {
            'variables': variables,
            'first_order': first_indices,
            'total_order': total_indices,
            'confidence': confidence,
            'output_name': output_name
        }

        return plot_data






