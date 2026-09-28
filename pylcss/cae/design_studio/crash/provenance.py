# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Immutable provenance manifests for reproducible crash simulation labels."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def mesh_fingerprint(mesh: Any) -> dict[str, object]:
    """Hash source coordinates/connectivity without embedding large arrays."""
    points = np.asarray(getattr(mesh, "p", []))
    connectivity = np.asarray(getattr(mesh, "t", []))
    digest = hashlib.sha256()
    for array in (points, connectivity):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii", errors="replace"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    shell_thickness = getattr(mesh, "shell_thickness", None)
    shell_nip = getattr(mesh, "shell_nip", None)
    digest.update(repr(shell_thickness).encode("ascii", errors="replace"))
    digest.update(repr(shell_nip).encode("ascii", errors="replace"))
    return {
        "sha256": digest.hexdigest(),
        "point_count": int(points.shape[1]) if points.ndim == 2 else 0,
        "element_count": (
            int(connectivity.shape[1]) if connectivity.ndim == 2 else 0
        ),
        "coordinate_shape": list(points.shape),
        "connectivity_shape": list(connectivity.shape),
        "shell_thickness_mm": (
            float(shell_thickness) if shell_thickness is not None else None
        ),
        "shell_integration_points": (
            int(shell_nip) if shell_nip is not None else None
        ),
    }


def build_crash_provenance(
    *,
    mesh: Any,
    material: Mapping[str, object],
    impact: Mapping[str, object],
    constraints: list[Mapping[str, object]],
    solver_settings: Mapping[str, object],
    deck_path: str | Path,
    engine_path: str | Path,
    starter_executable: str | None,
    engine_executable: str | None,
    time_history_converter: str | None,
    result_artifacts: Mapping[str, str | Path | None] | None = None,
) -> dict[str, object]:
    """Create a run manifest and a stable fingerprint of physics inputs."""
    physics_inputs = {
        "mesh": mesh_fingerprint(mesh),
        "material": _json_safe(material),
        "impact": _json_safe(impact),
        "constraints": _json_safe(constraints),
        "solver_settings": _json_safe(solver_settings),
    }
    canonical = json.dumps(
        physics_inputs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    run_id = hashlib.sha256(canonical).hexdigest()
    artifacts = {
        "starter_deck": {
            "path": str(Path(deck_path).resolve()),
            "sha256": sha256_file(deck_path),
        },
        "engine_deck": {
            "path": str(Path(engine_path).resolve()),
            "sha256": sha256_file(engine_path),
        },
    }
    for name, artifact_path in (result_artifacts or {}).items():
        artifacts[str(name)] = {
            "path": (
                str(Path(artifact_path).resolve())
                if artifact_path
                else None
            ),
            "sha256": sha256_file(artifact_path),
        }
    return {
        "schema": "pylcss.crash.provenance",
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physics_inputs": physics_inputs,
        "artifacts": artifacts,
        "executables": {
            "starter": {
                "path": starter_executable,
                "sha256": sha256_file(starter_executable),
            },
            "engine": {
                "path": engine_executable,
                "sha256": sha256_file(engine_executable),
            },
            "th_to_csv": {
                "path": time_history_converter,
                "sha256": sha256_file(time_history_converter),
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
    }


def write_crash_manifest(
    work_dir: str | Path,
    provenance: Mapping[str, object],
    quality: Mapping[str, object],
    metrics: Mapping[str, object],
) -> Path:
    """Write the compact audit record atomically next to solver artifacts."""
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "pylcss_crash_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    payload = {
        "provenance": _json_safe(provenance),
        "quality": _json_safe(quality),
        "metrics": _json_safe(metrics),
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
