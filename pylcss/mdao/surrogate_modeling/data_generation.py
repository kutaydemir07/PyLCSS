# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Safe, deterministic sampling of generated system-model functions."""

from __future__ import annotations

import importlib.util
import logging
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contracts import ProgressCallback, SpyPort, StopFlag

logger = logging.getLogger(__name__)

try:
    from scipy.stats import qmc

    QMC_AVAILABLE = True
except ImportError:
    QMC_AVAILABLE = False

# Kept as compatibility probes for clients that displayed these flags.
JOBLIB_AVAILABLE = importlib.util.find_spec("joblib") is not None
DILL_AVAILABLE = importlib.util.find_spec("dill") is not None

SpyFunction = Callable[..., tuple[Mapping[str, Any], Mapping[str, Any]]]
PortLike = SpyPort | str


def _port_names(ports: Sequence[PortLike], *, kind: str) -> list[str]:
    names: list[str] = []
    for index, port in enumerate(ports):
        if isinstance(port, str):
            name = port
        elif isinstance(port, Mapping):
            name = port.get("name")
        else:
            name = None
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{kind} port {index} has no non-empty string name.")
        names.append(name.strip())
    if len(names) != len(set(names)):
        raise ValueError(f"{kind} port names must be unique.")
    return names


def _validate_bounds(
    bounds: Sequence[tuple[float, float]],
    *,
    expected: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(bounds) != expected:
        raise ValueError(f"Expected {expected} input bounds, received {len(bounds)}.")
    array = np.asarray(bounds, dtype=np.float64)
    if array.shape != (expected, 2):
        raise ValueError("Each input bound must be a (lower, upper) pair.")
    if not np.isfinite(array).all():
        raise ValueError("Input bounds contain NaN or infinite values.")
    lower, upper = array[:, 0], array[:, 1]
    if np.any(lower > upper):
        bad = int(np.flatnonzero(lower > upper)[0])
        raise ValueError(
            f"Input bound {bad} has lower={lower[bad]} above upper={upper[bad]}."
        )
    return lower, upper


def _compile_spy_model(source: str) -> SpyFunction:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("spy_code must be a non-empty string.")
    module_name = f"_pylcss_spy_{uuid.uuid4().hex}"
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(Path.cwd() / f"{module_name}.py"),
        "__name__": module_name,
        "__package__": None,
    }
    try:
        compiled = compile(source, f"<{module_name}>", "exec")
        exec(compiled, namespace)
    except Exception as exc:
        logger.error("Generated spy-model code could not be compiled.", exc_info=True)
        logger.debug("Generated spy-model source:\n%s", source)
        raise RuntimeError(f"Failed to compile spy model: {exc}") from exc

    function = namespace.get("spy_model")
    if not callable(function):
        raise RuntimeError("Generated code must define a callable named 'spy_model'.")
    return function


def _sample_bounds(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    count: int,
    random_state: int,
) -> NDArray[np.float64]:
    if QMC_AVAILABLE:
        unit = qmc.LatinHypercube(d=lower.size, seed=random_state).random(n=count)
        return np.asarray(qmc.scale(unit, lower, upper), dtype=np.float64)
    rng = np.random.default_rng(random_state)
    return lower + rng.random((count, lower.size)) * (upper - lower)


def _evaluate_sample(
    spy_model: SpyFunction,
    sample: NDArray[np.float64],
    *,
    input_count: int,
    output_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    inputs, outputs = spy_model(*sample)
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise TypeError("spy_model must return a pair of input/output mappings.")
    input_values = np.asarray(
        [inputs[f"input_{index}"] for index in range(input_count)],
        dtype=np.float64,
    )
    output_values = np.asarray(
        [outputs[f"output_{index}"] for index in range(output_count)],
        dtype=np.float64,
    )
    if input_values.shape != (input_count,) or output_values.shape != (output_count,):
        raise ValueError("Every spy-model port must produce one scalar value.")
    if not np.isfinite(input_values).all() or not np.isfinite(output_values).all():
        raise ValueError("spy_model produced NaN or infinite values.")
    return input_values, output_values


def evaluate_spy_points(
    spy_code: str,
    spy_inputs: Sequence[PortLike],
    spy_outputs: Sequence[PortLike],
    points: np.ndarray,
    *,
    callback: ProgressCallback | None = None,
    stop_flag: StopFlag | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Evaluate an explicit adaptive batch without fabricating failed rows.

    Solver/CAD failures are reported and omitted from the returned arrays.  A
    zero response would create an artificial discontinuity and attract future
    active-learning rounds, so failed simulations must never be imputed here.
    """

    input_names = _port_names(spy_inputs, kind="Input")
    output_names = _port_names(spy_outputs, kind="Output")
    samples = np.asarray(points, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[1] != len(input_names):
        raise ValueError(
            "Adaptive points must be a two-dimensional array with one column "
            "per spy-model input."
        )

    spy_model = _compile_spy_model(spy_code)
    valid_X: list[NDArray[np.float64]] = []
    valid_y: list[NDArray[np.float64]] = []
    failures: list[str] = []
    total = len(samples)
    for index, sample in enumerate(samples):
        if stop_flag and stop_flag():
            break
        try:
            features, targets = _evaluate_sample(
                spy_model,
                sample,
                input_count=len(input_names),
                output_count=len(output_names),
            )
        except Exception as exc:
            message = f"Candidate {index} failed ({type(exc).__name__}): {exc}"
            failures.append(message)
            logger.warning(message)
        else:
            valid_X.append(features)
            valid_y.append(targets)

        if callback:
            callback(
                int(100 * (index + 1) / max(1, total)),
                f"Evaluated {index + 1}/{total} candidates "
                f"({len(failures)} failed)...",
            )

    features = (
        np.vstack(valid_X)
        if valid_X
        else np.empty((0, len(input_names)), dtype=np.float64)
    )
    targets = (
        np.vstack(valid_y)
        if valid_y
        else np.empty((0, len(output_names)), dtype=np.float64)
    )
    return features, targets, failures


def generate_spy_data(
    spy_code: str,
    spy_inputs: Sequence[PortLike],
    spy_outputs: Sequence[PortLike],
    input_bounds: Sequence[tuple[float, float]],
    *,
    num_samples: int = 1_000,
    test_samples: int = 200,
    random_state: int = 42,
    callback: ProgressCallback | None = None,
    stop_flag: StopFlag | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    list[str],
    list[str],
]:
    """Evaluate a generated system model over a Latin-hypercube design."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if test_samples < 0:
        raise ValueError("test_samples cannot be negative.")

    input_names = _port_names(spy_inputs, kind="Input")
    output_names = _port_names(spy_outputs, kind="Output")
    if not input_names:
        raise ValueError("At least one spy-model input is required.")
    if not output_names:
        raise ValueError("At least one spy-model output is required.")
    lower, upper = _validate_bounds(input_bounds, expected=len(input_names))

    if callback:
        callback(0, "Compiling generated model...")
    spy_model = _compile_spy_model(spy_code)
    requested = num_samples + test_samples
    samples = _sample_bounds(
        lower,
        upper,
        count=requested,
        random_state=random_state,
    )
    if callback:
        callback(10, f"Evaluating {requested} design samples...")

    valid_X: list[NDArray[np.float64]] = []
    valid_y: list[NDArray[np.float64]] = []
    consecutive_failures = 0
    max_consecutive_failures = 10
    last_error = ""

    # Generated engineering graphs frequently call non-thread-safe CAD/solver
    # runtimes. Sequential evaluation is deliberate and avoids Windows worker
    # crashes and unpicklable dynamically generated functions.
    for index, sample in enumerate(samples):
        if stop_flag and stop_flag():
            raise RuntimeError("Data generation was cancelled.")
        try:
            features, targets = _evaluate_sample(
                spy_model,
                sample,
                input_count=len(input_names),
                output_count=len(output_names),
            )
        except Exception as exc:
            consecutive_failures += 1
            last_error = f"{exc}\n{traceback.format_exc()}"
            logger.warning("Sample %d failed: %s", index, exc)
            logger.debug("Sample %d traceback:\n%s", index, last_error)
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    "Data generation stopped after "
                    f"{max_consecutive_failures} consecutive failures. "
                    f"Last error: {exc}"
                ) from exc
        else:
            valid_X.append(features)
            valid_y.append(targets)
            consecutive_failures = 0

        if callback and (
            index % max(1, requested // 20) == 0 or index == requested - 1
        ):
            progress = 10 + int(70 * (index + 1) / requested)
            callback(
                progress,
                f"Evaluated {index + 1}/{requested}; {len(valid_X)} valid samples.",
            )

    if len(valid_X) < 2:
        detail = f" Last error: {last_error.splitlines()[0]}" if last_error else ""
        raise RuntimeError(
            "Data generation produced fewer than two valid samples." + detail
        )

    features = np.vstack(valid_X)
    targets = np.vstack(valid_y)
    permutation = np.random.default_rng(random_state).permutation(features.shape[0])
    features = features[permutation]
    targets = targets[permutation]

    training_count = min(num_samples, features.shape[0])
    testing_count = min(test_samples, features.shape[0] - training_count)
    train_X = features[:training_count]
    train_y = targets[:training_count]
    test_X = features[training_count : training_count + testing_count]
    test_y = targets[training_count : training_count + testing_count]

    if training_count < num_samples:
        logger.warning(
            "Only %d of %d requested training samples were valid.",
            training_count,
            num_samples,
        )
    if callback:
        callback(80, "Data generation completed.")
    return train_X, train_y, test_X, test_y, input_names, output_names


__all__ = [
    "DILL_AVAILABLE",
    "JOBLIB_AVAILABLE",
    "QMC_AVAILABLE",
    "evaluate_spy_points",
    "generate_spy_data",
]
