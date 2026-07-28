# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Typed bridge between saved Design Studio studies and system models.

The Design Studio runtime already knows how to evaluate a saved ``.cad``
graph.  This module owns the higher-level contract needed by the Modeling
Environment: discover a study's public inputs and results, validate a selected
interface, and generate the managed function-block code.

There are intentionally no Qt imports here.  Keeping discovery and code
generation independent from the dialog makes the bridge testable headlessly
and gives future API/CLI integrations the exact same behaviour as the GUI.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
import keyword
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from pylcss.design_studio.runtime import discover_override_controls


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PARAM_SLOT_RE = re.compile(r"^param_(\d+)_name$")


@dataclass(frozen=True)
class StudyAnalysis:
    """One solver available in a saved Design Studio graph."""

    kind: str
    label: str
    node_id: str
    node_name: str


@dataclass(frozen=True)
class StudyInput:
    """A value that can be driven from a system-model input port."""

    name: str
    label: str
    source_kind: str
    target: str
    default: float
    lower: float
    upper: float
    unit: str
    group: str
    node_name: str = ""
    selected_by_default: bool = False


@dataclass(frozen=True)
class StudyOutput:
    """A standardized scalar result exposed by a Design Studio solver."""

    field: str
    label: str
    unit: str
    selected_by_default: bool = False


@dataclass(frozen=True)
class StudyDescriptor:
    """Complete public interface discovered from a saved ``.cad`` study."""

    path: str
    title: str
    analyses: tuple[StudyAnalysis, ...]
    inputs: tuple[StudyInput, ...]
    outputs: Mapping[str, tuple[StudyOutput, ...]]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimulationInputSpec:
    """A selected input port and its Design Studio target."""

    port_name: str
    label: str
    target_kind: str
    target: str
    default: float
    lower: float
    upper: float
    unit: str = "-"


@dataclass(frozen=True)
class SimulationOutputSpec:
    """A selected output port and its standardized result field."""

    port_name: str
    label: str
    result_field: str
    unit: str = "-"


@dataclass(frozen=True)
class SimulationFunctionSpec:
    """Serializable definition for a managed Design Studio function node."""

    project_path: str
    analysis_kind: str
    node_name: str
    inputs: tuple[SimulationInputSpec, ...]
    outputs: tuple[SimulationOutputSpec, ...]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationFunctionSpec":
        return cls(
            project_path=str(data["project_path"]),
            analysis_kind=str(data["analysis_kind"]),
            node_name=str(data["node_name"]),
            inputs=tuple(SimulationInputSpec(**item) for item in data.get("inputs", ())),
            outputs=tuple(SimulationOutputSpec(**item) for item in data.get("outputs", ())),
            version=int(data.get("version", 1)),
        )

    @classmethod
    def from_json(cls, text: str) -> "SimulationFunctionSpec":
        return cls.from_dict(json.loads(text))


ANALYSIS_LABELS = {
    "fea": "Static FEA",
    "crash": "Crash simulation",
    "topopt": "Topology optimization",
}

ANALYSIS_OUTPUTS: Mapping[str, tuple[StudyOutput, ...]] = {
    "fea": (
        StudyOutput("max_stress", "Maximum von Mises stress", "MPa", True),
        StudyOutput("peak_disp", "Maximum displacement", "mm", True),
        StudyOutput("compliance", "Compliance", "N·mm"),
        StudyOutput("strain_energy", "Strain energy", "N·mm"),
        StudyOutput("mass", "Mass", "tonne", True),
        StudyOutput("volume", "Volume", "mm³"),
    ),
    "crash": (
        StudyOutput("max_stress", "Maximum stress", "MPa", True),
        StudyOutput("peak_disp", "Maximum displacement", "mm", True),
        StudyOutput("absorbed_energy", "Absorbed energy", "N·mm", True),
        StudyOutput("peak_force", "Peak crushing force", "kN", True),
        StudyOutput("mean_force", "Mean crushing force", "kN", True),
        StudyOutput("crush_force_efficiency", "Crush force efficiency", "-", False),
        StudyOutput(
            "specific_energy_absorption",
            "Specific energy absorption",
            "kJ/kg",
            False,
        ),
        StudyOutput("crush_distance", "Useful crush distance", "mm", False),
        StudyOutput(
            "peak_acceleration_g",
            "Peak crash-pulse acceleration",
            "g",
            True,
        ),
        StudyOutput("delta_v", "Crash-pulse velocity change", "m/s", False),
        StudyOutput("n_failed", "Failed element count", "-", False),
    ),
    "topopt": (
        StudyOutput("final_vol_frac", "Final material fraction", "-", True),
        StudyOutput("compliance", "Compliance", "N·mm", True),
        StudyOutput("mass", "Optimized mass", "tonne", True),
        StudyOutput("volume", "Retained volume", "mm³"),
        StudyOutput("total_volume", "Original volume", "mm³"),
    ),
}


def is_python_identifier(name: str) -> bool:
    """Return whether *name* is safe as a generated Python variable."""

    return bool(_IDENTIFIER_RE.fullmatch(str(name or ""))) and not keyword.iskeyword(name)


def sanitize_identifier(value: str, fallback: str = "value") -> str:
    """Create a readable Python identifier while preserving common CAD names."""

    text = re.sub(r"\W+", "_", str(value or "").strip(), flags=re.UNICODE)
    text = re.sub(r"^(?=\d)", "_", text).strip("_")
    if not text:
        text = fallback
    if keyword.iskeyword(text):
        text += "_value"
    if not is_python_identifier(text):
        text = re.sub(r"[^A-Za-z0-9_]", "_", text)
        if not text or text[0].isdigit():
            text = f"{fallback}_{text}"
    return text


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_extra_parameters(raw: Any) -> dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}
    parsed: Mapping[str, Any] = {}
    if text.startswith("{"):
        try:
            candidate = ast.literal_eval(text)
            parsed = candidate if isinstance(candidate, Mapping) else {}
        except (SyntaxError, ValueError):
            parsed = {}
    else:
        values: dict[str, Any] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if is_python_identifier(name.strip()):
                values[name.strip()] = value.strip()
        parsed = values
    result: dict[str, float] = {}
    for name, value in parsed.items():
        number = _finite_number(value)
        if is_python_identifier(str(name)) and number is not None:
            result[str(name)] = number
    return result


def _suggest_bounds(value: float, name: str, group: str) -> tuple[float, float]:
    """Return conservative editable bounds for an imported design variable."""

    lower_name = name.lower()
    if lower_name in {"poissons_ratio", "volfrac", "density_cutoff"}:
        return max(0.0, value * 0.8), min(0.499 if "poisson" in lower_name else 1.0, value * 1.2)
    if value == 0.0:
        span = 1.0
        if group in {"Load", "Impact"}:
            span = 10.0
        return -span, span
    if value > 0.0:
        return value * 0.8, value * 1.2
    return value * 1.2, value * 0.8


def _infer_unit(name: str, group: str) -> str:
    key = name.lower()
    if key in {"poissons_ratio", "failure_strain", "volfrac", "density_cutoff", "tol"}:
        return "-"
    if any(token in key for token in ("enabled", "sensitive")):
        return "-"
    if key == "density":
        return "tonne/mm³"
    if key in {"youngs_modulus", "yield_strength", "tangent_modulus", "pressure", "stress_constraint", "yield_stress"}:
        return "MPa"
    if "velocity" in key:
        return "mm/s"
    if key == "gravity_accel":
        return "mm/s²"
    if key == "end_time":
        return "s"
    if key == "impactor_mass_kg":
        return "kg"
    if key.startswith("force_") or key == "force_magnitude":
        return "N"
    if group == "Geometry" or any(
        token in key for token in ("length", "width", "height", "thickness", "radius", "diameter", "gap", "size")
    ):
        return "mm"
    return "-"


def _analysis_kind(type_name: str) -> str | None:
    value = str(type_name or "")
    if value == "com.cad.sim.solver" or value.startswith("com.cad.sim.solver."):
        return "fea"
    if value == "com.cad.sim.crash_solver" or value.startswith("com.cad.sim.crash_solver."):
        return "crash"
    if value == "com.cad.sim.topopt_voxel" or value.startswith("com.cad.sim.topopt_voxel."):
        return "topopt"
    return None


def _unique_port_name(preferred: str, used: set[str]) -> str:
    base = sanitize_identifier(preferred)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def inspect_design_studio_study(path: str | Path) -> StudyDescriptor:
    """Inspect a saved Design Studio file without loading or executing its graph."""

    project_path = Path(path).expanduser().resolve()
    if not project_path.is_file():
        raise FileNotFoundError(f"Design Studio study not found: {project_path}")
    if project_path.suffix.lower() != ".cad":
        raise ValueError("Design Studio studies must use the .cad file extension.")

    try:
        session = json.loads(project_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Design Studio study JSON: {exc}") from exc
    nodes = session.get("nodes", {})
    if not isinstance(nodes, Mapping):
        raise ValueError("Invalid Design Studio study: missing node graph.")

    analyses: list[StudyAnalysis] = []
    parameters: dict[str, StudyInput] = {}
    warnings: list[str] = []

    for node_id, node_data in nodes.items():
        if not isinstance(node_data, Mapping):
            continue
        node_type = str(node_data.get("type_", ""))
        node_name = str(node_data.get("name") or "Design Studio node")
        custom = node_data.get("custom", {}) or {}
        if not isinstance(custom, Mapping):
            custom = {}

        kind = _analysis_kind(node_type)
        if kind:
            analyses.append(
                StudyAnalysis(kind, ANALYSIS_LABELS[kind], str(node_id), node_name)
            )

        exposed_name = str(custom.get("exposed_name", "") or "").strip()
        if not exposed_name and "variable" in node_type.lower():
            exposed_name = str(custom.get("variable_name", "") or "").strip()
        if exposed_name:
            default = _finite_number(custom.get("value_input", custom.get("value")))
            if default is not None:
                lower, upper = _suggest_bounds(default, exposed_name, "Geometry")
                parameters.setdefault(
                    exposed_name,
                    StudyInput(
                        name=exposed_name,
                        label=exposed_name,
                        source_kind="parameter",
                        target=exposed_name,
                        default=default,
                        lower=lower,
                        upper=upper,
                        unit=_infer_unit(exposed_name, "Geometry"),
                        group="Geometry",
                        node_name=node_name,
                        selected_by_default=True,
                    ),
                )

        slot_names: set[str] = set()
        for prop, name_value in custom.items():
            match = _PARAM_SLOT_RE.fullmatch(str(prop))
            parameter_name = str(name_value or "").strip()
            if not match or not parameter_name:
                continue
            if not is_python_identifier(parameter_name):
                warnings.append(
                    f"{node_name}: parameter {parameter_name!r} is not a valid Python name and was skipped."
                )
                continue
            slot_names.add(parameter_name)
            default = _finite_number(custom.get(f"param_{match.group(1)}_value"))
            if default is None:
                continue
            lower, upper = _suggest_bounds(default, parameter_name, "Geometry")
            parameters.setdefault(
                parameter_name,
                StudyInput(
                    name=parameter_name,
                    label=parameter_name,
                    source_kind="parameter",
                    target=parameter_name,
                    default=default,
                    lower=lower,
                    upper=upper,
                    unit=_infer_unit(parameter_name, "Geometry"),
                    group="Geometry",
                    node_name=node_name,
                    selected_by_default=True,
                ),
            )

        for parameter_name, default in _parse_extra_parameters(custom.get("parameters")).items():
            if parameter_name in slot_names:
                continue
            lower, upper = _suggest_bounds(default, parameter_name, "Geometry")
            parameters.setdefault(
                parameter_name,
                StudyInput(
                    name=parameter_name,
                    label=parameter_name,
                    source_kind="parameter",
                    target=parameter_name,
                    default=default,
                    lower=lower,
                    upper=upper,
                    unit=_infer_unit(parameter_name, "Geometry"),
                    group="Geometry",
                    node_name=node_name,
                    selected_by_default=True,
                ),
            )

    controls: list[StudyInput] = []
    used_names = set(parameters)
    for control in discover_override_controls(session):
        default = _finite_number(control.get("value"))
        if default is None:
            continue
        group = str(control.get("group") or "Analysis setting")
        prop = str(control.get("property") or "setting")
        preferred = prop
        if preferred in used_names:
            preferred = f"{sanitize_identifier(control.get('node', 'node'))}_{prop}"
        port_name = _unique_port_name(preferred, used_names)
        lower, upper = _suggest_bounds(default, prop, group)
        controls.append(
            StudyInput(
                name=port_name,
                label=f"{control.get('node', '')} · {control.get('label', prop)}",
                source_kind="setting",
                target=str(control["key"]),
                default=default,
                lower=lower,
                upper=upper,
                unit=_infer_unit(prop, group),
                group=group,
                node_name=str(control.get("node") or ""),
                selected_by_default=False,
            )
        )

    kind_order = {"fea": 0, "crash": 1, "topopt": 2}
    analyses.sort(key=lambda item: (kind_order.get(item.kind, 99), item.node_name))
    if not analyses:
        warnings.append("No FEA, crash, or topology-optimization solver was found.")
    elif len({item.kind for item in analyses}) != len(analyses):
        warnings.append(
            "The study contains more than one solver of the same type; the runtime uses the farthest downstream result."
        )

    inputs = tuple(parameters.values()) + tuple(
        sorted(controls, key=lambda item: (item.group, item.node_name, item.label))
    )
    available_outputs = {
        kind: ANALYSIS_OUTPUTS[kind]
        for kind in dict.fromkeys(item.kind for item in analyses)
    }
    return StudyDescriptor(
        path=str(project_path),
        title=str(session.get("_title") or project_path.stem).strip(),
        analyses=tuple(analyses),
        inputs=inputs,
        outputs=available_outputs,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def validate_simulation_spec(spec: SimulationFunctionSpec) -> None:
    """Raise ``ValueError`` when a selected node interface is unsafe."""

    project = Path(spec.project_path)
    if not project.is_file():
        raise ValueError(f"Design Studio study not found: {project}")
    if spec.analysis_kind not in ANALYSIS_OUTPUTS:
        raise ValueError(f"Unsupported Design Studio analysis: {spec.analysis_kind!r}")
    if not spec.inputs:
        raise ValueError("Select at least one Design Studio input.")
    if not spec.outputs:
        raise ValueError("Select at least one simulation result.")

    input_names = [item.port_name for item in spec.inputs]
    output_names = [item.port_name for item in spec.outputs]
    for name in input_names + output_names:
        if not is_python_identifier(name):
            raise ValueError(f"{name!r} is not a valid Python port name.")
    duplicates = {
        name for name in input_names + output_names
        if (input_names + output_names).count(name) > 1
    }
    if duplicates:
        raise ValueError(f"Port names must be unique: {', '.join(sorted(duplicates))}")

    targets = []
    for item in spec.inputs:
        if item.target_kind not in {"parameter", "setting"}:
            raise ValueError(f"Unsupported input target kind: {item.target_kind!r}")
        if item.target_kind == "parameter" and not is_python_identifier(item.target):
            raise ValueError(f"{item.target!r} is not a valid Design Studio parameter.")
        if item.target_kind == "setting" and "::" not in item.target:
            raise ValueError(f"{item.target!r} is not a valid Design Studio setting key.")
        if not all(math.isfinite(value) for value in (item.default, item.lower, item.upper)):
            raise ValueError(f"Input {item.port_name!r} has a non-finite value or bound.")
        if item.lower >= item.upper:
            raise ValueError(f"Input {item.port_name!r} needs lower bound < upper bound.")
        if not item.lower <= item.default <= item.upper:
            raise ValueError(f"Default for {item.port_name!r} must be inside its bounds.")
        target_key = (item.target_kind, item.target)
        if target_key in targets:
            raise ValueError(f"Design Studio target {item.target!r} is mapped more than once.")
        targets.append(target_key)

    valid_fields = {item.field for item in ANALYSIS_OUTPUTS[spec.analysis_kind]}
    for item in spec.outputs:
        if item.result_field not in valid_fields:
            raise ValueError(
                f"{item.result_field!r} is not a standard {spec.analysis_kind} result."
            )


def generate_simulation_function_code(spec: SimulationFunctionSpec) -> str:
    """Generate managed function-block statements for *spec*."""

    validate_simulation_spec(spec)
    parameter_lines: list[str] = []
    setting_lines: list[str] = []
    for item in spec.inputs:
        if item.target_kind == "parameter":
            parameter_lines.append(f"    {item.target}={item.port_name},")
        else:
            setting_lines.append(f"        {item.target!r}: {item.port_name},")

    path_text = str(Path(spec.project_path).resolve()).replace("\\", "/")
    lines = [
        "# Managed Design Studio Simulation node.",
        "# Refresh the node interface instead of editing this adapter by hand.",
        f"_study = cad.{spec.analysis_kind}(",
        f"    {path_text!r},",
    ]
    if setting_lines:
        lines.append("    _settings={")
        lines.extend(setting_lines)
        lines.append("    },")
    lines.extend(parameter_lines)
    lines.append(")")
    for item in spec.outputs:
        lines.append(f"{item.port_name} = _study.{item.result_field}")
    return "\n".join(lines) + "\n"


def make_default_spec(
    descriptor: StudyDescriptor,
    analysis_kind: str | None = None,
    *,
    node_name: str | None = None,
) -> SimulationFunctionSpec:
    """Build the default professional interface offered by the export dialog."""

    if not descriptor.analyses:
        raise ValueError("The Design Studio study has no runnable analysis.")
    selected_kind = analysis_kind or descriptor.analyses[0].kind
    if selected_kind not in descriptor.outputs:
        raise ValueError(f"The study does not contain a {selected_kind!r} analysis.")
    selected_inputs = tuple(
        SimulationInputSpec(
            port_name=item.name,
            label=item.label,
            target_kind=item.source_kind,
            target=item.target,
            default=item.default,
            lower=item.lower,
            upper=item.upper,
            unit=item.unit,
        )
        for item in descriptor.inputs
        if item.selected_by_default
    )
    selected_outputs = tuple(
        SimulationOutputSpec(
            port_name=item.field,
            label=item.label,
            result_field=item.field,
            unit=item.unit,
        )
        for item in descriptor.outputs[selected_kind]
        if item.selected_by_default
    )
    spec = SimulationFunctionSpec(
        project_path=descriptor.path,
        analysis_kind=selected_kind,
        node_name=node_name or f"{descriptor.title} · {ANALYSIS_LABELS[selected_kind]}",
        inputs=selected_inputs,
        outputs=selected_outputs,
    )
    validate_simulation_spec(spec)
    return spec


def find_input(
    inputs: Iterable[StudyInput], *, source_kind: str, target: str
) -> StudyInput | None:
    """Small lookup helper used by the dialog and tests."""

    return next(
        (
            item for item in inputs
            if item.source_kind == source_kind and item.target == target
        ),
        None,
    )
