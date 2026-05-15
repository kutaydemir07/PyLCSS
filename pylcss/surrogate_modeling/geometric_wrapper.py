# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
GeometryAwareWrapper -- joblib-friendly facade that lets a trained
:class:`GeomDeepONet` or :class:`GINONet` slot into the same
``surrogate.predict(X) -> (N, n_outputs)`` contract the tabular surrogates
use.

The wrapper's central trick is "live CAD per predict": when optimization
calls ``surrogate.predict([H, big_R, bolt_d])``, the wrapper drives the
PyLCSS CAD runtime at those parameters, gets back the corresponding mesh,
computes the SDF, runs the model, applies a user-chosen reduction
(max / mean / abs_max / ...) per output port, and returns scalars.

This is the only honest way to use geometric surrogates in design
optimization where parameters change the mesh.  It is also genuinely slow
compared to a tabular surrogate -- a CAD evaluation may take seconds.  The
:class:`GeometryCache` from :mod:`geometry` is wired in so repeated probes
of the same parameter vector are nearly free.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# Reductions reused from PhysicsNeMo-era code; the operation set is the same.
GEOMETRIC_REDUCTIONS: Dict[str, Any] = {
    "max":      lambda a: float(np.max(a)),
    "min":      lambda a: float(np.min(a)),
    "mean":     lambda a: float(np.mean(a)),
    "sum":      lambda a: float(np.sum(a)),
    "abs_max":  lambda a: float(np.max(np.abs(a))),
    "rms":      lambda a: float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2))),
}


try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    import torch.nn as nn

    class GeometryAwareWrapper:
        """
        Wraps a geometry-aware backbone (Geom-DeepONet or GINO) and exposes a
        tabular ``predict(X)`` API.

        Parameters
        ----------
        model : nn.Module
            Trained backbone.
        backbone : {"geom_deeponet", "gino"}
            Tells the wrapper which forward-pass adapter to use.
        cad_path : str
            CAD graph file driven by :mod:`pylcss.cad.runtime` to materialise
            geometry per design.
        cad_kind : {"fea", "crash", "topopt"}
            Which terminal solver to call.  Geometry only is used at predict
            time, so this is mostly about which graph branch to evaluate.
        input_param_names : list[str]
            Names of the design parameters in the order ``predict(X)``
            expects them.
        output_mapping : list[(field_name, reduction_op)]
            One entry per scalar output port the wrapper exposes downstream.
        field_widths : dict[str, int]
            Per-field component counts (e.g. ``{"von_mises": 1}``); the model
            outputs are concatenated along the channel axis in the same
            order as :attr:`field_names`.
        field_names : list[str]
            Ordered list of the fields the model was trained on.
        param_scaler_mean, param_scaler_std : np.ndarray
            Per-parameter mean/std used to normalize ``X`` before passing to
            the model.  Stored on the wrapper because the model itself only
            sees normalized inputs.
        grid_size : int, optional
            Background-grid resolution for GINO.  Ignored for Geom-DeepONet.
        """

        def __init__(
            self,
            model: nn.Module,
            backbone: str,
            cad_path: str,
            cad_kind: str,
            input_param_names: List[str],
            output_mapping: List[Tuple[str, str]],
            field_widths: Dict[str, int],
            field_names: List[str],
            param_scaler_mean: np.ndarray,
            param_scaler_std: np.ndarray,
            grid_size: int = 32,
        ) -> None:
            self.model = model
            self.backbone = backbone
            self.cad_path = cad_path
            self.cad_kind = cad_kind
            self.input_param_names = list(input_param_names)
            self.output_mapping = list(output_mapping)
            self.field_widths = dict(field_widths)
            self.field_names = list(field_names)
            self.param_scaler_mean = np.asarray(param_scaler_mean, dtype=np.float64)
            self.param_scaler_std = np.asarray(param_scaler_std, dtype=np.float64)
            self.grid_size = int(grid_size)
            self.device = next(model.parameters()).device

        # ------------------------------------------------------------------
        # The interface optimization / sensitivity / solution-space code uses.
        # ------------------------------------------------------------------
        def predict(
            self, X: np.ndarray, return_std: bool = False,
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            X = np.asarray(X, dtype=np.float64)
            squeeze_out = False
            if X.ndim == 1:
                X = X.reshape(1, -1)
                squeeze_out = True

            n_rows = X.shape[0]
            n_out = len(self.output_mapping)
            results = np.zeros((n_rows, n_out), dtype=np.float64)

            # Per-field slice into the model's stacked output channels.
            slices: Dict[str, slice] = {}
            offset = 0
            for name in self.field_names:
                w = max(1, int(self.field_widths.get(name, 1)))
                slices[name] = slice(offset, offset + w)
                offset += w

            for r in range(n_rows):
                results[r] = self._predict_one(X[r], slices)

            if squeeze_out:
                results = results[0]
                if return_std:
                    return results, np.zeros_like(results)
                return results
            if return_std:
                return results, np.zeros_like(results)
            return results

        # ------------------------------------------------------------------
        # The actual per-row pipeline. Heavy lifting happens here.
        # ------------------------------------------------------------------
        def _predict_one(self, x_row: np.ndarray, slices: Dict[str, slice]) -> np.ndarray:
            from pylcss.surrogate_modeling.geometry import (
                evaluate_with_cache, compute_sdf, make_background_grid,
                TRIMESH_AVAILABLE,
            )

            if not TRIMESH_AVAILABLE:
                raise RuntimeError(
                    "trimesh is required at predict time for SDF computation. "
                    "Install with: pip install trimesh"
                )

            params = {
                name: float(val)
                for name, val in zip(self.input_param_names, x_row)
            }

            # 1. Live CAD evaluation: gives us the actual mesh at this design.
            geom = evaluate_with_cache(
                self.cad_path, self.cad_kind, params, field_name=None,
            )

            # 2. Normalize params for the model.
            params_norm = (np.asarray(x_row, dtype=np.float64) - self.param_scaler_mean) / np.where(
                self.param_scaler_std > 1e-12, self.param_scaler_std, 1.0
            )
            params_t = torch.as_tensor(params_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                if self.backbone == "geom_deeponet":
                    # Query points: the design's own mesh nodes + their SDF.
                    pts = geom.points  # (n_nodes, 3)
                    sdf = compute_sdf(geom.points, geom.cells, pts)  # all zero on surface, by construction
                    query = np.column_stack([pts, sdf]).astype(np.float32)
                    query_t = torch.as_tensor(query, device=self.device)
                    field = self.model(params_t, query_t)  # (1, n_nodes, out_dim)
                    field_np = field.cpu().numpy()[0]
                elif self.backbone == "gino":
                    # Background grid: rasterise SDF on a fixed-size grid; sample
                    # model output back at the design's mesh nodes.
                    bbox_min, bbox_max = geom.bbox
                    grid_pts, grid_shape = make_background_grid(
                        bbox_min, bbox_max, resolution=self.grid_size,
                    )
                    grid_sdf = compute_sdf(geom.points, geom.cells, grid_pts)
                    R = grid_shape[0]
                    sdf_volume = torch.as_tensor(
                        grid_sdf.reshape(1, 1, R, R, R), dtype=torch.float32, device=self.device,
                    )
                    # Normalize query coords into [-1, 1] over the *padded* bbox.
                    extent = bbox_max - bbox_min
                    pad = extent * 0.1
                    lo, hi = bbox_min - pad, bbox_max + pad
                    span = hi - lo
                    span = np.where(span > 1e-12, span, 1.0)
                    pts_norm = 2.0 * (geom.points - lo) / span - 1.0
                    # grid_sample expects (..., z, y, x) ordering
                    pts_norm = pts_norm[:, [2, 1, 0]].astype(np.float32)
                    query_t = torch.as_tensor(pts_norm, device=self.device)
                    field = self.model(sdf_volume, params_t, query_t)  # (1, n_nodes, out_dim)
                    field_np = field.cpu().numpy()[0]
                else:
                    raise ValueError(f"Unknown backbone: {self.backbone!r}")

            # 3. Reduction: collapse the (n_nodes, total_out_channels) field to
            #    one scalar per (field_name, op) entry in output_mapping.
            out = np.zeros(len(self.output_mapping), dtype=np.float64)
            for col, (fname, op) in enumerate(self.output_mapping):
                sl = slices.get(fname)
                if sl is None:
                    slab = field_np
                else:
                    stop = min(sl.stop, field_np.shape[1])
                    start = min(sl.start, stop)
                    slab = field_np[:, start:stop]
                reducer = GEOMETRIC_REDUCTIONS.get(op, GEOMETRIC_REDUCTIONS["max"])
                out[col] = reducer(slab)
            return out

        # joblib-friendly load/save: store on CPU so the file is portable.
        def to_cpu(self) -> "GeometryAwareWrapper":
            self.model.cpu()
            self.device = torch.device("cpu")
            return self

else:
    class GeometryAwareWrapper:
        pass
