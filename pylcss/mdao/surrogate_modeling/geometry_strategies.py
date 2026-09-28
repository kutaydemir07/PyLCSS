# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Geom-DeepONet and compact GINO training implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .contracts import LossCallback, ProgressCallback, StopFlag
from .geometry_data import (
    GeometrySample,
    GeometryScaling,
    GeometryStrategyBase,
    positive_float,
    positive_int,
    select_torch_device,
)
from .geometry_estimator import GeometryAwareWrapper
from .geometry_loop import fit_geometry_model
from .operators import GeomDeepONet, GINONet
from .spatial import compute_sdf, make_background_grid, normalize_grid_coordinates
from .strategies import TrainingResult


class GeomDeepONetStrategy(GeometryStrategyBase):
    """Train a point-cloud Geom-DeepONet on varying CAD meshes."""

    backbone_name = "geom_deeponet"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        callback: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
        loss_callback: LossCallback | None = None,
    ) -> TrainingResult:
        del X, y, X_test, y_test
        (
            cad_path,
            cad_kind,
            field_name,
            input_names,
            input_bounds,
            n_samples,
            epochs,
        ) = self.validate_config(config)
        random_state = int(config.get("random_state", 42))
        torch.manual_seed(random_state)
        samples = self.collect_samples(
            cad_path,
            cad_kind,
            field_name,
            input_names,
            input_bounds,
            n_samples,
            random_state,
            callback,
            stop_flag,
        )
        training, validation, evaluation_source = self.split_samples(
            samples,
            validation_fraction=float(config.get("validation_split", 0.2)),
            random_state=random_state,
        )
        scaling = self.scaling(training)
        output_dim = training[0].field.shape[1]
        device = select_torch_device()
        model = GeomDeepONet(
            n_param=len(input_names),
            out_dim=output_dim,
            latent_dim=positive_int(config, "latent_dim", 64),
            branch_hidden=positive_int(config, "branch_hidden", 128),
            branch_layers=positive_int(config, "branch_layers", 3),
            trunk_hidden=positive_int(config, "trunk_hidden", 64),
            trunk_layers=positive_int(config, "trunk_layers", 4),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=positive_float(config, "learning_rate", 1e-3),
        )
        criterion = torch.nn.MSELoss()

        prepared = [self._prepare_sample(sample, scaling) for sample in training]
        prepared_validation = [
            self._prepare_sample(sample, scaling) for sample in validation
        ]
        loss_history = fit_geometry_model(
            model,
            optimizer,
            prepared,
            prepared_validation,
            loss_function=lambda sample: self._loss(
                model,
                sample,
                criterion,
                device,
            ),
            epochs=epochs,
            random_state=random_state,
            callback=callback,
            stop_flag=stop_flag,
            loss_callback=loss_callback,
            patience=positive_int(config, "patience", 50),
        )
        actual, predicted = self._predict(
            model,
            prepared_validation,
            scaling,
            device,
        )
        metrics = self.finish_metrics(
            actual,
            predicted,
            backbone=self.backbone_name,
            sample_count=len(samples),
            source=evaluation_source,
            loss_history=loss_history,
        )
        wrapper = GeometryAwareWrapper(
            model=model,
            backbone=self.backbone_name,
            cad_path=cad_path,
            cad_kind=cad_kind,
            input_param_names=input_names,
            output_mapping=[(field_name, "max")],
            field_widths={field_name: output_dim},
            field_names=[field_name],
            param_scaler_mean=scaling.parameter_mean,
            param_scaler_std=scaling.parameter_std,
            field_scaler_mean=scaling.field_mean,
            field_scaler_std=scaling.field_std,
            coordinate_center=scaling.coordinate_center,
            coordinate_scale=scaling.coordinate_scale,
        )
        return wrapper, metrics

    @staticmethod
    def _prepare_sample(
        sample: GeometrySample,
        scaling: GeometryScaling,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = (
            (sample.parameters - scaling.parameter_mean) / scaling.parameter_std
        ).astype(np.float32)
        sdf = compute_sdf(
            sample.geometry.points,
            sample.geometry.cells,
            sample.geometry.points,
        )
        points = (
            sample.geometry.points - scaling.coordinate_center
        ) / scaling.coordinate_scale
        query = np.column_stack([points, sdf / scaling.coordinate_scale]).astype(
            np.float32
        )
        field = ((sample.field - scaling.field_mean) / scaling.field_std).astype(
            np.float32
        )
        return parameters, query, field

    @staticmethod
    def _loss(
        model: GeomDeepONet,
        sample: tuple[np.ndarray, np.ndarray, np.ndarray],
        criterion: torch.nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        parameters, query, field = sample
        parameters_t = torch.as_tensor(
            parameters,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        query_t = torch.as_tensor(query, dtype=torch.float32, device=device)
        target_t = torch.as_tensor(
            field,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        return criterion(model(parameters_t, query_t), target_t)

    @staticmethod
    def _predict(
        model: GeomDeepONet,
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        scaling: GeometryScaling,
        device: torch.device,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        actual: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for parameters, query, field in samples:
                parameters_t = torch.as_tensor(
                    parameters,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                query_t = torch.as_tensor(
                    query,
                    dtype=torch.float32,
                    device=device,
                )
                result = model(parameters_t, query_t)[0].detach().cpu().numpy()
                actual.append(field * scaling.field_std + scaling.field_mean)
                predicted.append(result * scaling.field_std + scaling.field_mean)
        return actual, predicted


class GINOStrategy(GeometryStrategyBase):
    """Train the compact SDF-grid/FNO geometry operator used by PyLCSS."""

    backbone_name = "gino"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Mapping[str, Any],
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        callback: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
        loss_callback: LossCallback | None = None,
    ) -> TrainingResult:
        del X, y, X_test, y_test
        (
            cad_path,
            cad_kind,
            field_name,
            input_names,
            input_bounds,
            n_samples,
            epochs,
        ) = self.validate_config(config)
        random_state = int(config.get("random_state", 42))
        torch.manual_seed(random_state)
        grid_size = positive_int(config, "grid_size", 24)
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4 for spectral convolution.")
        samples = self.collect_samples(
            cad_path,
            cad_kind,
            field_name,
            input_names,
            input_bounds,
            n_samples,
            random_state,
            callback,
            stop_flag,
        )
        training, validation, evaluation_source = self.split_samples(
            samples,
            validation_fraction=float(config.get("validation_split", 0.2)),
            random_state=random_state,
        )
        scaling = self.scaling(training)
        output_dim = training[0].field.shape[1]
        device = select_torch_device()
        model = GINONet(
            n_param=len(input_names),
            out_dim=output_dim,
            grid_size=grid_size,
            modes=positive_int(config, "fno_modes", 4),
            hidden_channels=positive_int(config, "hidden_channels", 16),
            n_fno_layers=positive_int(config, "fno_layers", 3),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=positive_float(config, "learning_rate", 1e-3),
        )
        criterion = torch.nn.MSELoss()
        prepared = [
            self._prepare_sample(sample, scaling, grid_size) for sample in training
        ]
        prepared_validation = [
            self._prepare_sample(sample, scaling, grid_size) for sample in validation
        ]
        loss_history = fit_geometry_model(
            model,
            optimizer,
            prepared,
            prepared_validation,
            loss_function=lambda sample: self._loss(
                model,
                sample,
                criterion,
                device,
                grid_size,
            ),
            epochs=epochs,
            random_state=random_state,
            callback=callback,
            stop_flag=stop_flag,
            loss_callback=loss_callback,
            patience=positive_int(config, "patience", 50),
        )
        actual, predicted = self._predict(
            model,
            prepared_validation,
            scaling,
            device,
            grid_size,
        )
        metrics = self.finish_metrics(
            actual,
            predicted,
            backbone=self.backbone_name,
            sample_count=len(samples),
            source=evaluation_source,
            loss_history=loss_history,
        )
        wrapper = GeometryAwareWrapper(
            model=model,
            backbone=self.backbone_name,
            cad_path=cad_path,
            cad_kind=cad_kind,
            input_param_names=input_names,
            output_mapping=[(field_name, "max")],
            field_widths={field_name: output_dim},
            field_names=[field_name],
            param_scaler_mean=scaling.parameter_mean,
            param_scaler_std=scaling.parameter_std,
            grid_size=grid_size,
            field_scaler_mean=scaling.field_mean,
            field_scaler_std=scaling.field_std,
        )
        return wrapper, metrics

    @staticmethod
    def _prepare_sample(
        sample: GeometrySample,
        scaling: GeometryScaling,
        grid_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        parameters = (
            (sample.parameters - scaling.parameter_mean) / scaling.parameter_std
        ).astype(np.float32)
        bbox_min, bbox_max = sample.geometry.bbox
        grid_points, _ = make_background_grid(
            bbox_min,
            bbox_max,
            resolution=grid_size,
        )
        grid_sdf = compute_sdf(
            sample.geometry.points,
            sample.geometry.cells,
            grid_points,
        )
        query, distance_scale = normalize_grid_coordinates(
            sample.geometry.points,
            bbox_min,
            bbox_max,
        )
        field = ((sample.field - scaling.field_mean) / scaling.field_std).astype(
            np.float32
        )
        return (
            parameters,
            (grid_sdf / distance_scale).astype(np.float32),
            query,
            field,
        )

    @staticmethod
    def _loss(
        model: GINONet,
        sample: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        criterion: torch.nn.Module,
        device: torch.device,
        grid_size: int,
    ) -> torch.Tensor:
        parameters, grid_sdf, query, field = sample
        parameters_t = torch.as_tensor(
            parameters,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        sdf_t = torch.as_tensor(
            grid_sdf.reshape(1, 1, grid_size, grid_size, grid_size),
            dtype=torch.float32,
            device=device,
        )
        query_t = torch.as_tensor(query, dtype=torch.float32, device=device)
        target_t = torch.as_tensor(
            field,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        return criterion(model(sdf_t, parameters_t, query_t), target_t)

    @staticmethod
    def _predict(
        model: GINONet,
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        scaling: GeometryScaling,
        device: torch.device,
        grid_size: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        actual: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for parameters, grid_sdf, query, field in samples:
                parameters_t = torch.as_tensor(
                    parameters,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                sdf_t = torch.as_tensor(
                    grid_sdf.reshape(1, 1, grid_size, grid_size, grid_size),
                    dtype=torch.float32,
                    device=device,
                )
                query_t = torch.as_tensor(
                    query,
                    dtype=torch.float32,
                    device=device,
                )
                result = model(sdf_t, parameters_t, query_t)[0].detach().cpu().numpy()
                actual.append(field * scaling.field_std + scaling.field_mean)
                predicted.append(result * scaling.field_std + scaling.field_mean)
        return actual, predicted


__all__ = ["GINOStrategy", "GeomDeepONetStrategy"]
