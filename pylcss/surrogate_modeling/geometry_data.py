# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""CAD sampling and normalization shared by geometry-training strategies."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cad_geometry import CadGeometry, cad_evaluate_geometry
from .contracts import Metrics, ProgressCallback, StopFlag
from .data_generation import QMC_AVAILABLE
from .metrics import evaluate_predictions
from .spatial import TRIMESH_AVAILABLE
from .strategies import SurrogateModelStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeometrySample:
    """One evaluated design and its nodal target field."""

    parameters: np.ndarray
    geometry: CadGeometry
    field: np.ndarray


@dataclass(frozen=True)
class GeometryScaling:
    """Normalization statistics persisted with a trained geometry model."""

    parameter_mean: np.ndarray
    parameter_std: np.ndarray
    field_mean: np.ndarray
    field_std: np.ndarray
    coordinate_center: np.ndarray
    coordinate_scale: float


def select_torch_device() -> torch.device:
    """Select the best available device without assuming accelerator support."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    """Read a strictly positive integer configuration value."""
    value = int(config.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be at least 1.")
    return value


def positive_float(config: Mapping[str, Any], key: str, default: float) -> float:
    """Read a finite, strictly positive float configuration value."""
    value = float(config.get(key, default))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive.")
    return value


class GeometryStrategyBase(SurrogateModelStrategy):
    """Shared data collection, normalization, splitting, and evaluation."""

    backbone_name = "geometry"

    @staticmethod
    def validate_config(
        config: Mapping[str, Any],
    ) -> tuple[str, str, str, list[str], list[tuple[float, float]], int, int]:
        """Validate common geometry-training settings."""
        if not TRIMESH_AVAILABLE:
            raise RuntimeError("trimesh is required for geometry-aware surrogates.")

        cad_path = str(config.get("cad_path", "")).strip()
        if not cad_path:
            raise ValueError("cad_path is required for geometry-aware training.")
        if not Path(cad_path).is_file():
            raise FileNotFoundError(f"CAD graph does not exist: {cad_path}")

        cad_kind = str(config.get("cad_kind", "fea")).lower()
        if cad_kind not in {"fea", "impact", "crash", "topopt"}:
            raise ValueError("cad_kind must be 'fea', 'impact', or 'topopt'.")
        field_name = str(config.get("field_name", "")).strip()
        if not field_name:
            raise ValueError("field_name is required for geometry-aware training.")

        raw_names = config.get("input_names")
        raw_bounds = config.get("input_bounds")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, str | bytes):
            raise ValueError("input_names must be a sequence of names.")
        if not isinstance(raw_bounds, Sequence) or isinstance(raw_bounds, str | bytes):
            raise ValueError("input_bounds must be a sequence of (lower, upper) pairs.")
        input_names = [str(name).strip() for name in raw_names]
        if not input_names or any(not name for name in input_names):
            raise ValueError("input_names must contain non-empty names.")
        if len(input_names) != len(set(input_names)):
            raise ValueError("input_names must be unique.")

        bounds_array = np.asarray(raw_bounds, dtype=np.float64)
        if bounds_array.shape != (len(input_names), 2):
            raise ValueError("input_bounds must contain one pair per input name.")
        if not np.isfinite(bounds_array).all():
            raise ValueError("input_bounds contains NaN or infinite values.")
        if np.any(bounds_array[:, 0] > bounds_array[:, 1]):
            raise ValueError("Every lower input bound must be <= its upper bound.")
        input_bounds = [(float(low), float(high)) for low, high in bounds_array]

        n_samples = positive_int(config, "n_samples", 30)
        epochs = positive_int(config, "epochs", 500)
        return (
            cad_path,
            cad_kind,
            field_name,
            input_names,
            input_bounds,
            n_samples,
            epochs,
        )

    @staticmethod
    def sample_parameters(
        bounds: Sequence[tuple[float, float]],
        *,
        count: int,
        random_state: int,
    ) -> np.ndarray:
        """Create a deterministic Latin-hypercube parameter design."""
        lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
        upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
        if QMC_AVAILABLE:
            from scipy.stats import qmc

            unit = qmc.LatinHypercube(d=len(bounds), seed=random_state).random(count)
            return np.asarray(qmc.scale(unit, lower, upper), dtype=np.float64)
        rng = np.random.default_rng(random_state)
        return lower + rng.random((count, len(bounds))) * (upper - lower)

    def collect_samples(
        self,
        cad_path: str,
        cad_kind: str,
        field_name: str,
        input_names: Sequence[str],
        input_bounds: Sequence[tuple[float, float]],
        n_samples: int,
        random_state: int,
        callback: ProgressCallback | None,
        stop_flag: StopFlag | None,
    ) -> list[GeometrySample]:
        """Evaluate CAD/solver graphs and retain consistent nodal fields."""
        parameters = self.sample_parameters(
            input_bounds,
            count=n_samples,
            random_state=random_state,
        )
        samples: list[GeometrySample] = []
        output_width: int | None = None
        for index, row in enumerate(parameters):
            if stop_flag and stop_flag():
                raise RuntimeError("Geometry data collection was cancelled.")
            values = {
                name: float(value) for name, value in zip(input_names, row, strict=True)
            }
            try:
                geometry = cad_evaluate_geometry(
                    cad_path,
                    cad_kind,
                    values,
                    field_name=field_name,
                )
                field = geometry.fields.get(field_name)
                if field is None:
                    available = ", ".join(sorted(geometry.fields)) or "none"
                    raise ValueError(
                        f"Nodal field {field_name!r} is missing; available fields: "
                        f"{available}."
                    )
                field = np.asarray(field, dtype=np.float64)
                if field.ndim == 1:
                    field = field.reshape(-1, 1)
                if field.ndim != 2 or field.shape[0] != geometry.n_nodes:
                    raise ValueError(
                        f"Field {field_name!r} has shape {field.shape}; expected "
                        f"({geometry.n_nodes}, n_components)."
                    )
                if not np.isfinite(field).all():
                    raise ValueError(
                        f"Field {field_name!r} contains non-finite values."
                    )
                if output_width is None:
                    output_width = field.shape[1]
                elif field.shape[1] != output_width:
                    raise ValueError(
                        f"Field width changed from {output_width} to {field.shape[1]}."
                    )
            except Exception as exc:
                logger.warning(
                    "CAD sample %d/%d failed for %s: %s",
                    index + 1,
                    n_samples,
                    values,
                    exc,
                )
            else:
                samples.append(GeometrySample(row.copy(), geometry, field))

            if callback:
                callback(
                    5 + int(55 * (index + 1) / n_samples),
                    f"CAD sample {index + 1}/{n_samples}; {len(samples)} valid.",
                )

        if len(samples) < 2:
            raise RuntimeError(
                "Geometry training requires at least two successful CAD evaluations."
            )
        return samples

    @staticmethod
    def split_samples(
        samples: list[GeometrySample],
        *,
        validation_fraction: float,
        random_state: int,
    ) -> tuple[list[GeometrySample], list[GeometrySample], str]:
        """Create a design-level holdout without leaking mesh nodes."""
        if len(samples) < 5 or not 0.0 < validation_fraction < 1.0:
            logger.warning(
                "Too few geometry samples for a reliable holdout; "
                "reporting training-set metrics."
            )
            return samples, samples, "training"
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(samples))
        validation_count = max(1, round(len(samples) * validation_fraction))
        validation_count = min(validation_count, len(samples) - 2)
        validation_indices = set(indices[:validation_count].tolist())
        training = [
            sample
            for index, sample in enumerate(samples)
            if index not in validation_indices
        ]
        validation = [
            sample
            for index, sample in enumerate(samples)
            if index in validation_indices
        ]
        return training, validation, "holdout"

    @staticmethod
    def scaling(samples: Sequence[GeometrySample]) -> GeometryScaling:
        """Fit parameter, field, and coordinate scaling on training designs."""
        parameters = np.vstack([sample.parameters for sample in samples])
        parameter_mean = parameters.mean(axis=0)
        parameter_std = parameters.std(axis=0)
        parameter_std = np.where(parameter_std > 1e-12, parameter_std, 1.0)

        fields = np.vstack([sample.field for sample in samples])
        field_mean = fields.mean(axis=0)
        field_std = fields.std(axis=0)
        field_std = np.where(field_std > 1e-12, field_std, 1.0)

        bbox_min = np.min(
            np.vstack([sample.geometry.bbox[0] for sample in samples]),
            axis=0,
        )
        bbox_max = np.max(
            np.vstack([sample.geometry.bbox[1] for sample in samples]),
            axis=0,
        )
        coordinate_center = (bbox_min + bbox_max) / 2.0
        coordinate_scale = max(float(np.max(bbox_max - bbox_min)) / 2.0, 1e-12)
        return GeometryScaling(
            parameter_mean,
            parameter_std,
            field_mean,
            field_std,
            coordinate_center,
            coordinate_scale,
        )

    @staticmethod
    def finish_metrics(
        actual: list[np.ndarray],
        predicted: list[np.ndarray],
        *,
        backbone: str,
        sample_count: int,
        source: str,
        loss_history: list[float],
    ) -> Metrics:
        """Build metrics over complete held-out nodal fields."""
        metrics = evaluate_predictions(np.vstack(actual), np.vstack(predicted))
        metrics.update(
            {
                "debug_mode": False,
                "backbone": backbone,
                "n_samples": sample_count,
                "evaluation_source": source,
                "loss_history": loss_history,
            }
        )
        return metrics


__all__ = [
    "GeometrySample",
    "GeometryScaling",
    "GeometryStrategyBase",
    "positive_float",
    "positive_int",
    "select_torch_device",
]
