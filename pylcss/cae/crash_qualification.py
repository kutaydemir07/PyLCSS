# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Project policies and metadata checks for physical crash benchmarks.

These checks inspect supplied metadata; they do not authenticate source documents
or recompute declared file hashes. Curve comparison lives in crash_validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Sequence


DEFAULT_QUALIFICATION_METADATA = (
    "benchmark_id",
    "specimen_id",
    "test_date",
    "test_laboratory",
    "geometry_revision",
    "material_lot_id",
    "boundary_condition",
    "trigger_definition",
    "time_zero_definition",
    "source_document",
    "source_document_sha256",
    "csv_sha256",
    "metadata_sha256",
    "license_or_permission",
    "approved_by",
    "approval_date",
)

BENCHMARK_CHANNELS = ("force_time", "crash_pulse", "force_displacement")


def _finite_number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not measurements.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("A finite number is required.")
    return number


def _positive_integer(value: object) -> int:
    number = _finite_number(value)
    if number < 1 or not number.is_integer():
        raise ValueError("A positive integer is required.")
    return int(number)


@dataclass(frozen=True)
class QualificationPolicy:
    """Configure documentation and acquisition requirements for a project.

    Set required_metadata to the project's documentation fields, omitting approval
    fields when approval paperwork is not required. An empty positive_conventions
    mapping disables convention-label checks. Curves must already use compatible
    units and signs; this validator does not transform them. A zero sample-rate
    ratio disables that ratio check, while finite positive acquisition values are
    still required.
    """

    impact_condition_tolerance: float = 0.01
    sample_rate_cfc_ratio: float = 10.0
    required_metadata: Sequence[str] = DEFAULT_QUALIFICATION_METADATA
    positive_conventions: Mapping[str, str] = field(
        default_factory=lambda: {
            "force_channel": "compression_positive",
            "acceleration_channel": "deceleration_positive",
            "displacement_channel": "crush_positive",
        }
    )
    require_matching_filters: bool = True
    require_traceability_pass: bool = True

    def __post_init__(self) -> None:
        for name in ("impact_condition_tolerance", "sample_rate_cfc_ratio"):
            value = _finite_number(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        if isinstance(self.required_metadata, str) or any(
            not isinstance(name, str) or not name.strip()
            for name in self.required_metadata
        ):
            raise ValueError("required_metadata must contain field names.")
        allowed = {"force_channel", "acceleration_channel", "displacement_channel"}
        if not isinstance(self.positive_conventions, Mapping) or any(
            name not in allowed or not isinstance(value, str) or not value.strip()
            for name, value in self.positive_conventions.items()
        ):
            raise ValueError("positive_conventions must map channels to sign labels.")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _present(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    text = str(value).strip()
    return bool(text) and text.upper() not in {"REQUIRED", "REPLACE_WITH_TEST_ID"}


def validate_benchmark_metadata(
    simulation: Mapping[str, object],
    benchmark: Mapping[str, object],
    *,
    required_channels: Sequence[str],
    minimum_replicates: int = 3,
    policy: QualificationPolicy | None = None,
) -> dict[str, object]:
    """Validate only the metadata needed by the selected benchmark channels.

    Positive thickness and an integer replicate count are always required. Impact
    mass matching applies to inertial scenarios, not a prescribed moving wall.
    """
    if not required_channels or set(required_channels) - set(BENCHMARK_CHANNELS):
        raise ValueError("Select at least one supported benchmark channel.")
    minimum_replicates = _positive_integer(minimum_replicates)
    policy = policy if policy is not None else QualificationPolicy()
    traceability = _mapping(benchmark.get("traceability"))
    measurement = _mapping(simulation.get("measurement"))
    processing = _mapping(simulation.get("processing"))
    failures = []

    if policy.require_traceability_pass and str(
        traceability.get("status", "")
    ).lower() != "pass":
        failures.append("traceability status is not pass")
    for name in policy.required_metadata:
        if not _present(traceability.get(name)):
            failures.append(f"missing or placeholder {name}")

    selected = []
    if {"force_time", "force_displacement"}.intersection(required_channels):
        selected.append(("force_channel", "force_filter", ()))
    if "crash_pulse" in required_channels:
        selected.append(
            ("acceleration_channel", "acceleration_filter", ("location", "axis"))
        )
    if "force_displacement" in required_channels:
        selected.append(("displacement_channel", None, ()))

    for name, filter_name, extra_fields in selected:
        channel = _mapping(traceability.get(name))
        for key in ("sensor_id", "calibration_id", *extra_fields):
            if not _present(channel.get(key)):
                failures.append(f"missing or placeholder {name}.{key}")
        expected_convention = policy.positive_conventions.get(name)
        if expected_convention is not None and channel.get(
            "positive_convention"
        ) != expected_convention:
            failures.append(f"{name} positive convention must be {expected_convention}")

        numbers = {}
        numeric_fields = (
            ("sample_rate_hz", "filter_cfc") if filter_name else ("sample_rate_hz",)
        )
        for key in numeric_fields:
            try:
                number = _finite_number(channel.get(key))
                if number <= 0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                failures.append(f"{name}.{key} must be finite and positive")
            else:
                numbers[key] = number
        if filter_name:
            if len(numbers) == 2 and numbers["sample_rate_hz"] < (
                policy.sample_rate_cfc_ratio * numbers["filter_cfc"]
            ):
                failures.append(
                    f"{name} sample rate is below the "
                    f"{policy.sample_rate_cfc_ratio:g}x-CFC policy"
                )
            expected_cfc = _mapping(processing.get(filter_name)).get("cfc")
            if policy.require_matching_filters and expected_cfc is not None:
                try:
                    expected_cfc = _finite_number(expected_cfc)
                    if expected_cfc <= 0:
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    failures.append(
                        f"simulation {filter_name}.cfc must be finite and positive"
                    )
                else:
                    if numbers.get("filter_cfc") != expected_cfc:
                        failures.append(f"{name} CFC does not match simulation")

    impact_fields = [("impact_velocity_m_s", "initial_speed_m_s")]
    if measurement.get("scenario") != "prescribed_moving_wall":
        impact_fields.append(("impactor_mass_kg", "impactor_mass_kg"))
    for name, simulation_name in impact_fields:
        try:
            test_value = _finite_number(traceability.get(name))
            simulation_value = _finite_number(measurement.get(simulation_name))
            if min(test_value, simulation_value) < 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            failures.append(
                f"{name} must be finite and nonnegative in test and simulation"
            )
            continue
        error = abs(test_value - simulation_value) / max(
            abs(test_value), abs(simulation_value), 1.0e-12
        )
        if error > policy.impact_condition_tolerance:
            failures.append(
                f"{name} differs from simulation by more than "
                f"{policy.impact_condition_tolerance:.1%}"
            )
    try:
        if _finite_number(traceability.get("thickness_mm")) <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        failures.append("thickness_mm must be finite and positive")
    try:
        replicate_count = _positive_integer(traceability.get("test_replicate_count"))
    except (TypeError, ValueError, OverflowError):
        failures.append("test_replicate_count must be a positive integer")
    else:
        if replicate_count < minimum_replicates:
            failures.append(
                f"test_replicate_count must be at least {minimum_replicates}"
            )

    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "minimum_replicates": minimum_replicates,
        "policy": asdict(policy),
    }
