# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Sensitivity Analysis Module for PyLCSS.

This module provides global sensitivity analysis using multiple methods:
- Morris screening (elementary effects) for variable screening
- Sobol indices (variance-based) for quantitative sensitivity
- FAST (Fourier Amplitude Sensitivity Test) for efficient first-order
- Delta Moment-Independent Measure (DMIM) for distribution-based sensitivity

Supports multi-output batch analysis, S2 interaction matrices,
convergence analysis, and importance ranking.
"""

import warnings
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import SALib for sensitivity analysis
try:
    from SALib.sample import sobol, morris as morris_sample
    from SALib.analyze import sobol as sobol_analyze, morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

# Try FAST / DMIM (SALib >=1.4)
try:
    from SALib.sample import fast_sampler
    from SALib.analyze import fast as fast_analyze
    FAST_AVAILABLE = True
except ImportError:
    FAST_AVAILABLE = False

try:
    from SALib.analyze import delta as delta_analyze
    from SALib.sample import latin as latin_sample
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False


class SensitivityAnalyzer:
    """
    Performs global sensitivity analysis using multiple methods.

    Supported methods:
    - Morris (Screening): Quickly identify negligible variables.
    - Sobol (Variance-based): Full first-order, total-order, and second-order indices.
    - FAST (Fourier): Efficient first-order estimation.
    - Delta (Moment-Independent): Distribution-based sensitivity measure.
    """

    METHODS = ['Sobol', 'Morris', 'FAST', 'Delta']

    def __init__(self):
        if not SALIB_AVAILABLE:
            raise ImportError("SALib is required for sensitivity analysis. Install with: pip install SALib")

    @staticmethod
    def available_methods() -> List[str]:
        """Return list of available analysis methods based on installed packages."""
        methods = []
        if SALIB_AVAILABLE:
            methods.extend(['Sobol', 'Morris'])
        if FAST_AVAILABLE:
            methods.append('FAST')
        if DELTA_AVAILABLE:
            methods.append('Delta')
        return methods

    def _adjust_to_power_of_two(self, n: int) -> int:
        """Adjust n to the nearest power of 2 for optimal Sobol sequence convergence."""
        if n <= 0:
            return 2
        power = math.floor(math.log2(n))
        lower = 2 ** power
        upper = 2 ** (power + 1)
        return lower if abs(n - lower) <= abs(n - upper) else upper

    # ========================================================================
    # Morris Screening
    # ========================================================================

    def run_screening(self, problem_definition: Dict[str, Any],
                      n_trajectories: int = 20) -> Tuple[np.ndarray, Dict]:
        """
        Generate Morris screening samples.

        Args:
            problem_definition: Dict with 'names' and 'bounds'.
            n_trajectories: Number of trajectories (default 20).

        Returns:
            (samples_X, salib_problem)
        """
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }
        X = morris_sample.sample(problem, n_trajectories, num_levels=4)
        return X, problem

    def analyze_screening(self, X: np.ndarray, Y: np.ndarray,
                          problem: Dict[str, Any],
                          threshold_pct: float = 0.01) -> Tuple[List[str], Dict]:
        """
        Analyze Morris screening results.

        Args:
            X: Input samples.
            Y: Output values.
            problem: SALib problem definition.
            threshold_pct: Fraction of max(mu_star) for importance cut-off.

        Returns:
            (important_variable_names, morris_results_dict)
        """
        si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)

        mu_star = np.array(si['mu_star'])
        sigma = np.array(si['sigma'])
        mu = np.array(si['mu'])
        max_mu = np.max(mu_star) if np.max(mu_star) > 0 else 1.0
        threshold = threshold_pct * max_mu

        important_indices = np.where(mu_star > threshold)[0]
        important_vars = [problem['names'][i] for i in important_indices]
        if not important_vars:
            important_vars = list(problem['names'])

        results = {
            'variable_names': problem['names'],
            'mu': mu.tolist(),
            'mu_star': mu_star.tolist(),
            'sigma': sigma.tolist(),
            'mu_star_conf': si.get('mu_star_conf', np.zeros_like(mu_star)).tolist() if hasattr(si, 'get') else [],
            'important_variables': important_vars,
            'method': 'Morris'
        }
        return important_vars, results

    # ========================================================================
    # Sobol Indices
    # ========================================================================

    def generate_samples(self, problem_definition: Dict[str, Any],
                         n_samples: int = 1024) -> np.ndarray:
        """Generate Sobol sequence samples."""
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }
        X = sobol.sample(problem, n_samples)
        return X

    def analyze_sensitivity(self, X: np.ndarray, Y: np.ndarray,
                            problem_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Sobol sensitivity analysis.

        Returns dict with S1, ST, S2, confidence intervals, and interaction matrix.
        """
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }

        Si = sobol_analyze.analyze(problem, Y, print_to_console=False)

        n_vars = problem['num_vars']

        # Build S2 interaction matrix
        s2_matrix = np.zeros((n_vars, n_vars))
        s2_raw = Si.get('S2', None)
        if s2_raw is not None:
            s2_arr = np.array(s2_raw)
            if s2_arr.ndim == 2:
                s2_matrix = s2_arr.copy()
                # Make symmetric
                s2_matrix = np.where(np.isnan(s2_matrix), 0, s2_matrix)
                s2_matrix = s2_matrix + s2_matrix.T
                np.fill_diagonal(s2_matrix, np.array(Si['S1']))

        results = {
            'variable_names': problem_definition['names'],
            'first_order': np.array(Si['S1']),
            'total_order': np.array(Si['ST']),
            'second_order': s2_raw,
            'confidence_first': np.array(Si['S1_conf']),
            'confidence_total': np.array(Si['ST_conf']),
            's2_matrix': s2_matrix,
            'method': 'Sobol'
        }
        return results

    # ========================================================================
    # FAST Method
    # ========================================================================

    def generate_fast_samples(self, problem_definition: Dict[str, Any],
                               n_samples: int = 1024) -> np.ndarray:
        """Generate FAST samples."""
        if not FAST_AVAILABLE:
            raise ImportError("SALib FAST sampler not available.")
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }
        X = fast_sampler.sample(problem, n_samples)
        return X

    def analyze_fast(self, X: np.ndarray, Y: np.ndarray,
                     problem_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Perform FAST sensitivity analysis (first-order only)."""
        if not FAST_AVAILABLE:
            raise ImportError("SALib FAST analyzer not available.")
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }
        Si = fast_analyze.analyze(problem, Y, print_to_console=False)
        return {
            'variable_names': problem_definition['names'],
            'first_order': np.array(Si['S1']),
            'total_order': np.array(Si['ST']),
            'confidence_first': np.zeros(len(problem_definition['names'])),
            'confidence_total': np.zeros(len(problem_definition['names'])),
            'method': 'FAST'
        }

    # ========================================================================
    # Delta Moment-Independent Measure
    # ========================================================================

    def generate_delta_samples(self, problem_definition: Dict[str, Any],
                                n_samples: int = 1024) -> np.ndarray:
        """Generate Latin Hypercube samples for Delta analysis."""
        if not DELTA_AVAILABLE:
            raise ImportError("SALib Delta analyzer not available.")
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }
        X = latin_sample.sample(problem, n_samples)
        return X

    def analyze_delta(self, X: np.ndarray, Y: np.ndarray,
                      problem_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Delta Moment-Independent Measure analysis."""
        if not DELTA_AVAILABLE:
            raise ImportError("SALib Delta analyzer not available.")
        problem = {
            'num_vars': len(problem_definition['names']),
            'names': problem_definition['names'],
            'bounds': problem_definition['bounds']
        }
        Si = delta_analyze.analyze(problem, X, Y, print_to_console=False)
        return {
            'variable_names': problem_definition['names'],
            'delta': np.array(Si['delta']),
            'delta_conf': np.array(Si['delta_conf']),
            'S1': np.array(Si['S1']),
            'S1_conf': np.array(Si['S1_conf']),
            'method': 'Delta'
        }

    # ========================================================================
    # Multi-output batch analysis
    # ========================================================================

    def batch_analyze(self, X: np.ndarray, Y_dict: Dict[str, np.ndarray],
                      problem_definition: Dict[str, Any],
                      method: str = 'Sobol') -> Dict[str, Dict[str, Any]]:
        """
        Run sensitivity analysis for multiple output variables at once.

        Args:
            X: Input sample matrix (shared across all outputs).
            Y_dict: Dict mapping output_name -> output_values array.
            problem_definition: Problem definition dict.
            method: Analysis method ('Sobol', 'Morris', 'FAST', 'Delta').

        Returns:
            Dict mapping output_name -> analysis results.
        """
        all_results = {}
        for output_name, Y in Y_dict.items():
            try:
                if method == 'Sobol':
                    all_results[output_name] = self.analyze_sensitivity(X, Y, problem_definition)
                elif method == 'FAST':
                    all_results[output_name] = self.analyze_fast(X, Y, problem_definition)
                elif method == 'Delta':
                    all_results[output_name] = self.analyze_delta(X, Y, problem_definition)
                else:
                    logger.warning("batch_analyze: method '%s' not supported for batch. Using Sobol.", method)
                    all_results[output_name] = self.analyze_sensitivity(X, Y, problem_definition)
            except Exception as e:
                logger.error("Sensitivity analysis failed for output '%s': %s", output_name, e)
                all_results[output_name] = {'error': str(e)}
        return all_results

    # ========================================================================
    # Convergence Analysis
    # ========================================================================

    def convergence_analysis(self, problem_definition: Dict[str, Any],
                             evaluate_fn,
                             output_name: str,
                             sample_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run Sobol analysis at increasing sample sizes to assess convergence.

        Args:
            problem_definition: Problem definition dict.
            evaluate_fn: Callable(x_array) -> dict of outputs.
            output_name: Name of the output to track.
            sample_sizes: List of sample sizes (powers of 2). Default [64,128,256,512,1024].

        Returns:
            Dict with 'sample_sizes', 'S1_traces', 'ST_traces' (each n_sizes × n_vars).
        """
        if sample_sizes is None:
            sample_sizes = [64, 128, 256, 512, 1024]

        n_vars = len(problem_definition['names'])
        S1_traces = np.zeros((len(sample_sizes), n_vars))
        ST_traces = np.zeros((len(sample_sizes), n_vars))

        for idx, n_s in enumerate(sample_sizes):
            X = self.generate_samples(problem_definition, n_s)
            Y = np.zeros(len(X))
            for i, xi in enumerate(X):
                try:
                    result = evaluate_fn(xi)
                    Y[i] = float(result.get(output_name, 0.0))
                except Exception:
                    Y[i] = 0.0

            res = self.analyze_sensitivity(X, Y, problem_definition)
            S1_traces[idx] = res['first_order']
            ST_traces[idx] = res['total_order']

        return {
            'sample_sizes': sample_sizes,
            'variable_names': problem_definition['names'],
            'S1_traces': S1_traces,
            'ST_traces': ST_traces
        }

    # ========================================================================
    # Importance Ranking
    # ========================================================================

    @staticmethod
    def rank_variables(results: Dict[str, Any],
                       metric: str = 'total_order') -> List[Dict[str, Any]]:
        """
        Rank variables by sensitivity index and classify importance.

        Args:
            results: Analysis results dict.
            metric: Which index to rank by ('total_order', 'first_order').

        Returns:
            Sorted list of dicts with 'name', 'index', 'rank', 'category'.
        """
        names = results['variable_names']
        values = np.array(results.get(metric, results.get('total_order', np.zeros(len(names)))))

        sorted_idx = np.argsort(-values)  # descending
        ranked = []
        for rank, i in enumerate(sorted_idx, 1):
            val = float(values[i])
            if val > 0.2:
                category = 'Critical'
            elif val > 0.05:
                category = 'Important'
            elif val > 0.01:
                category = 'Minor'
            else:
                category = 'Negligible'

            ranked.append({
                'name': names[i],
                'index': val,
                'rank': rank,
                'category': category
            })
        return ranked

    # ========================================================================
    # Plot helpers
    # ========================================================================

    def plot_sensitivity_indices(self, results: Dict[str, Any],
                                 output_name: str = "Output") -> Dict[str, Any]:
        """Return plot data dict for PyQtGraph rendering."""
        return {
            'variables': results['variable_names'],
            'first_order': results.get('first_order', []),
            'total_order': results.get('total_order', []),
            'confidence': results.get('confidence_total', None),
            'output_name': output_name,
            'method': results.get('method', 'Sobol')
        }

