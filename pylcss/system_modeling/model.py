# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Executable system-model container and source loader."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any
from uuid import uuid4

from pylcss.config import TEMP_MODELS_DIR

from .types import CompiledModel, InputSpec, ModelCallable, OutputSpec

_GENERATED_FUNCTION = re.compile(r"system_function(?:_\d+)?")


class ModelSourceError(ValueError):
    """Raised when generated model source cannot provide one entry point."""


class SystemModel:
    """A callable system model plus its serializable interface metadata.

    Source passed to :meth:`from_code_string` is trusted application/user code.
    It is imported as Python and is not a security sandbox.
    """

    def __init__(
        self,
        name: str,
        system_function: ModelCallable,
        inputs: Sequence[Mapping[str, Any]],
        outputs: Sequence[Mapping[str, Any]],
        source_code: str | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must be a non-empty string.")
        if not callable(system_function):
            raise TypeError("system_function must be callable.")

        self.name = name.strip()
        self.system_function = system_function
        # Kept as an alias for callers written against earlier releases. Running
        # object-mode Numba here caused duplicate execution when a model raised.
        self.fast_function = system_function
        self.inputs = _copy_specs(inputs, "input")
        self.outputs = _copy_specs(outputs, "output")
        self.source_code = source_code

    @classmethod
    def from_code_string(
        cls,
        name: str,
        code_string: str,
        inputs: Sequence[Mapping[str, Any]],
        outputs: Sequence[Mapping[str, Any]],
    ) -> SystemModel:
        """Import trusted source and create a process-importable model."""

        if not isinstance(code_string, str) or not code_string.strip():
            raise ModelSourceError("Model source must be a non-empty string.")

        source_bytes = code_string.encode("utf-8")
        digest = hashlib.sha256(source_bytes).hexdigest()[:24]
        module_name = f"_pylcss_model_{digest}"
        model_dir = Path(TEMP_MODELS_DIR).resolve()
        model_path = model_dir / f"{module_name}.py"

        # Report syntax errors before touching the generated-model cache.
        compile(code_string, str(model_path), "exec")
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_source_once(model_path, code_string)

        model_dir_text = os.fspath(model_dir)
        if model_dir_text not in sys.path:
            sys.path.insert(0, model_dir_text)

        module = _load_module(module_name, model_path)
        system_function = _find_entry_point(module)
        return cls(name, system_function, inputs, outputs, code_string)

    @classmethod
    def from_models(
        cls,
        models: Sequence[CompiledModel],
        merged_name: str = "Merged",
    ) -> SystemModel:
        """Create one executable model from one or more compiled models."""

        if not models:
            raise ValueError("At least one model is required.")
        if len(models) == 1:
            model = models[0]
            return cls.from_code_string(
                model["name"],
                model["code"],
                model["inputs"],
                model["outputs"],
            )

        from .merge import create_merged_model

        merged = create_merged_model(models)
        return cls.from_code_string(
            merged_name,
            merged["code"],
            merged["inputs"],
            merged["outputs"],
        )

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Evaluate the model exactly once."""

        result = self.system_function(**kwargs)
        if not isinstance(result, Mapping):
            raise TypeError(
                f"Model {self.name!r} returned {type(result).__name__}; "
                "expected a mapping of output names to values."
            )
        return dict(result)

    def get_input_names(self) -> list[str]:
        return [item["name"] for item in self.inputs]

    def get_output_names(self) -> list[str]:
        return [item["name"] for item in self.outputs]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible model metadata and source."""

        return {
            "name": self.name,
            "inputs": [dict(item) for item in self.inputs],
            "outputs": [dict(item) for item in self.outputs],
            "source_code": self.source_code,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SystemModel:
        """Rebuild a model serialized by :meth:`to_dict`."""

        source = data.get("source_code")
        if not isinstance(source, str) or not source.strip():
            raise ModelSourceError("Serialized model does not contain source code.")
        return cls.from_code_string(
            str(data["name"]),
            source,
            data["inputs"],
            data["outputs"],
        )


def _copy_specs(
    specs: Sequence[Mapping[str, Any]],
    kind: str,
) -> list[InputSpec] | list[OutputSpec]:
    copied: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise TypeError(f"Model {kind} {index} must be a mapping.")
        item = dict(spec)
        variable_name = item.get("name")
        if not isinstance(variable_name, str) or not variable_name:
            raise ValueError(f"Model {kind} {index} has no valid name.")
        if variable_name in names:
            raise ValueError(f"Duplicate model {kind} name: {variable_name!r}.")
        names.add(variable_name)
        copied.append(item)
    return copied  # type: ignore[return-value]


def _write_source_once(path: Path, source: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != source:
            raise ModelSourceError(f"Generated-model cache collision at {path}.")
        return

    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(source, encoding="utf-8", newline="\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_module(module_name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModelSourceError(f"Cannot create an import specification for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _find_entry_point(module: ModuleType) -> ModelCallable:
    exact = getattr(module, "system_function", None)
    if callable(exact):
        return exact

    candidates: list[Callable[..., Any]] = [
        value
        for name, value in vars(module).items()
        if _GENERATED_FUNCTION.fullmatch(name) and callable(value)
    ]
    if len(candidates) != 1:
        raise ModelSourceError(
            "Model source must define 'system_function' or exactly one "
            "'system_function_<number>' entry point."
        )
    return candidates[0]


__all__ = ["ModelSourceError", "SystemModel"]




