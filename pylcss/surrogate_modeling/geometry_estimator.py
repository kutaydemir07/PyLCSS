# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
GeometryAwareWrapper is a joblib-friendly facade that lets a trained
:class:`GeomDeepONet` or :class:`GINONet` slot into the same
``surrogate.predict(X) -> (N, n_outputs)`` contract the tabular surrogates
use.

The wrapper performs a live CAD evaluation for each prediction: when optimization
calls ``surrogate.predict([H, big_R, bolt_d])``, the wrapper drives the
PyLCSS CAD runtime at those parameters, gets back the corresponding mesh,
computes the SDF, runs the model, applies a user-chosen reduction
(max / mean / abs_max / ...) per output port, and returns scalars.

This preserves mesh-changing parameter effects, but it remains much slower than
a tabular surrogate because a CAD evaluation may take seconds. Repeated probes
are served by :class:`GeometryCache`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Reductions reused from PhysicsNeMo-era code; the operation set is the same.
GEOMETRIC_REDUCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "max": lambda a: float(np.max(a)),
    "min": lambda a: float(np.min(a)),
    "mean": lambda a: float(np.mean(a)),
    "sum": lambda a: float(np.sum(a)),
    "abs_max": lambda a: float(np.max(np.abs(a))),
    "rms": lambda a: float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2))),
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
            CAD graph file driven by :mod:`pylcss.design_studio.runtime` to materialise
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
            input_param_names: Sequence[str],
            output_mapping: Sequence[tuple[str, str]],
            field_widths: Mapping[str, int],
            field_names: Sequence[str],
            param_scaler_mean: np.ndarray,
            param_scaler_std: np.ndarray,
            grid_size: int = 32,
            field_scaler_mean: np.ndarray | None = None,
            field_scaler_std: np.ndarray | None = None,
            coordinate_center: np.ndarray | None = None,
            coordinate_scale: float = 1.0,
        ) -> None:
            if backbone not in {"geom_deeponet", "gino"}:
                raise ValueError(f"Unknown geometry backbone: {backbone!r}.")
            if not input_param_names:
                raise ValueError("At least one input parameter is required.")
            if not output_mapping:
                raise ValueError("At least one output reduction is required.")
            invalid_reductions = {
                reduction
                for _, reduction in output_mapping
                if reduction not in GEOMETRIC_REDUCTIONS
            }
            if invalid_reductions:
                raise ValueError(
                    "Unknown output reductions: "
                    + ", ".join(sorted(invalid_reductions))
                )
            if grid_size < 2:
                raise ValueError("grid_size must be at least 2.")
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
            if self.param_scaler_mean.shape != (len(self.input_param_names),):
                raise ValueError("param_scaler_mean does not match input parameters.")
            if self.param_scaler_std.shape != (len(self.input_param_names),):
                raise ValueError("param_scaler_std does not match input parameters.")
            self.grid_size = int(grid_size)
            output_width = sum(
                max(1, self.field_widths.get(name, 1)) for name in self.field_names
            )
            self.field_scaler_mean: NDArray[np.float64] = np.zeros(
                output_width, dtype=np.float64
            )
            self.field_scaler_std: NDArray[np.float64] = np.ones(
                output_width, dtype=np.float64
            )
            if field_scaler_mean is not None:
                self.field_scaler_mean = np.asarray(field_scaler_mean, dtype=np.float64)
            if field_scaler_std is not None:
                self.field_scaler_std = np.asarray(field_scaler_std, dtype=np.float64)
            if self.field_scaler_mean.shape != (output_width,):
                raise ValueError("field_scaler_mean does not match model output width.")
            if self.field_scaler_std.shape != (output_width,):
                raise ValueError("field_scaler_std does not match model output width.")
            self.coordinate_center: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
            if coordinate_center is not None:
                self.coordinate_center = np.asarray(coordinate_center, dtype=np.float64)
            if self.coordinate_center.shape != (3,):
                raise ValueError("coordinate_center must contain three values.")
            if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
                raise ValueError("coordinate_scale must be finite and positive.")
            self.coordinate_scale = float(coordinate_scale)
            self.device = next(model.parameters()).device

        # ------------------------------------------------------------------
        # The interface optimization / sensitivity / solution-space code uses.
        # ------------------------------------------------------------------
        @overload
        def predict(
            self,
            X: np.ndarray,
            return_std: Literal[False] = False,
        ) -> np.ndarray: ...

        @overload
        def predict(
            self,
            X: np.ndarray,
            return_std: Literal[True],
        ) -> tuple[np.ndarray, np.ndarray]: ...

        def predict(
            self,
            X: np.ndarray,
            return_std: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
            X = np.asarray(X, dtype=np.float64)
            squeeze_out = False
            if X.ndim == 1:
                X = X.reshape(1, -1)
                squeeze_out = True
            if X.ndim != 2:
                raise ValueError(f"X must be one- or two-dimensional; got {X.shape}.")
            if X.shape[1] != len(self.input_param_names):
                raise ValueError(
                    f"Expected {len(self.input_param_names)} input parameters; "
                    f"received {X.shape[1]}."
                )
            if not np.isfinite(X).all():
                raise ValueError("X contains NaN or infinite values.")

            n_rows = X.shape[0]
            n_out = len(self.output_mapping)
            results = np.zeros((n_rows, n_out), dtype=np.float64)

            # Per-field slice into the model's stacked output channels.
            slices: dict[str, slice] = {}
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
        def _predict_one(
            self,
            x_row: np.ndarray,
            slices: dict[str, slice],
        ) -> np.ndarray:
            from pylcss.surrogate_modeling.geometry_cache import evaluate_with_cache
            from pylcss.surrogate_modeling.spatial import (
                TRIMESH_AVAILABLE,
                compute_sdf,
                make_background_grid,
                normalize_grid_coordinates,
            )

            if not TRIMESH_AVAILABLE:
                raise RuntimeError(
                    "trimesh is required at predict time for SDF computation. "
                    "Install with: pip install trimesh"
                )

            params = {
                name: float(val)
                for name, val in zip(self.input_param_names, x_row, strict=False)
            }

            # 1. Live CAD evaluation: gives us the actual mesh at this design.
            geom = evaluate_with_cache(
                self.cad_path,
                self.cad_kind,
                params,
                field_name=None,
            )

            # 2. Normalize params for the model.
            params_norm = (
                np.asarray(x_row, dtype=np.float64) - self.param_scaler_mean
            ) / np.where(self.param_scaler_std > 1e-12, self.param_scaler_std, 1.0)
            params_t = torch.as_tensor(
                params_norm, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                if self.backbone == "geom_deeponet":
                    # Query points: the design's own mesh nodes + their SDF.
                    pts = geom.points  # (n_nodes, 3)
                    sdf = compute_sdf(
                        geom.points, geom.cells, pts
                    )  # all zero on surface, by construction
                    center = np.asarray(
                        getattr(self, "coordinate_center", np.zeros(3)),
                        dtype=np.float64,
                    )
                    scale = float(getattr(self, "coordinate_scale", 1.0))
                    normalized_points = (pts - center) / scale
                    normalized_sdf = sdf / scale
                    query = np.column_stack([normalized_points, normalized_sdf]).astype(
                        np.float32
                    )
                    query_t = torch.as_tensor(query, device=self.device)
                    field = self.model(params_t, query_t)  # (1, n_nodes, out_dim)
                    field_np = field.cpu().numpy()[0]
                elif self.backbone == "gino":
                    # Background grid: rasterise SDF on a fixed-size grid; sample
                    # model output back at the design's mesh nodes.
                    bbox_min, bbox_max = geom.bbox
                    grid_pts, grid_shape = make_background_grid(
                        bbox_min,
                        bbox_max,
                        resolution=self.grid_size,
                    )
                    grid_sdf = compute_sdf(geom.points, geom.cells, grid_pts)
                    R = grid_shape[0]
                    pts_norm, distance_scale = normalize_grid_coordinates(
                        geom.points,
                        bbox_min,
                        bbox_max,
                    )
                    sdf_volume = torch.as_tensor(
                        (grid_sdf / distance_scale).reshape(1, 1, R, R, R),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    query_t = torch.as_tensor(pts_norm, device=self.device)
                    field = self.model(
                        sdf_volume, params_t, query_t
                    )  # (1, n_nodes, out_dim)
                    field_np = field.cpu().numpy()[0]
                else:
                    raise ValueError(f"Unknown backbone: {self.backbone!r}")

            field_mean = np.asarray(
                getattr(self, "field_scaler_mean", np.zeros(field_np.shape[1])),
                dtype=np.float64,
            )
            field_std = np.asarray(
                getattr(self, "field_scaler_std", np.ones(field_np.shape[1])),
                dtype=np.float64,
            )
            if field_mean.shape != (field_np.shape[1],) or field_std.shape != (
                field_np.shape[1],
            ):
                raise RuntimeError(
                    "Stored field scaling is incompatible with model output."
                )
            field_np = field_np * field_std + field_mean

            # 3. Reduction: collapse the (n_nodes, total_out_channels) field to
            #    one scalar per (field_name, op) entry in output_mapping.
            out: NDArray[np.float64] = np.zeros(
                len(self.output_mapping), dtype=np.float64
            )
            for col, (fname, op) in enumerate(self.output_mapping):
                sl = slices.get(fname)
                if sl is None:
                    raise RuntimeError(
                        f"Output field {fname!r} was not part of model training."
                    )
                stop = min(sl.stop, field_np.shape[1])
                start = min(sl.start, stop)
                slab = field_np[:, start:stop]
                if slab.size == 0:
                    raise RuntimeError(f"Output field {fname!r} has no model channels.")
                try:
                    reducer = GEOMETRIC_REDUCTIONS[op]
                except KeyError as exc:
                    raise ValueError(f"Unknown reduction operation: {op!r}.") from exc
                out[col] = reducer(slab)
            return out

        # joblib-friendly load/save: store on CPU so the file is portable.
        def to_cpu(self) -> GeometryAwareWrapper:
            self.model.cpu()
            self.device = torch.device("cpu")
            return self

else:

    class GeometryAwareWrapper:  # type: ignore[no-redef]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PyTorch is required for geometry-aware surrogates.")
