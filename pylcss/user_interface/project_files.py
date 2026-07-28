# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Application-project naming, manifest writing, and preflight validation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from pylcss.io_manager._atomic import PathLike
from pylcss.io_manager.project_io import atomic_json_dump, load_json_object

__all__ = [
    "ProjectValidationError",
    "save_project_manifest",
    "sanitize_project_name",
    "validate_project_folder",
]

_CURRENT_FORMAT_VERSION = 2
_INVALID_WINDOWS_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_JSON_COMPONENTS: dict[str, tuple[str, ...]] = {
    "design_studio.cad": ("nodes",),
    "surrogate_settings.json": (),
    "solution_space.json": ("version",),
    "optimization_setup.json": (
        "variables",
        "objectives",
        "constraints",
        "settings",
    ),
    "sensitivity.json": ("method",),
}


class ProjectValidationError(ValueError):
    """Raised when a folder is not a complete, supported PyLCSS project."""


def sanitize_project_name(name: str, *, fallback: str = "New_Project") -> str:
    """Return a portable directory name for a user-supplied product name."""
    cleaned = _INVALID_WINDOWS_NAME_CHARS.sub("_", str(name)).strip(" .")
    cleaned = cleaned or fallback
    if cleaned.upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_Project"
    return cleaned


def save_project_manifest(folder_path: PathLike) -> Path:
    """Atomically write the versioned application-project manifest."""
    folder = Path(folder_path).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "pylcss-project",
        "version": _CURRENT_FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "components": [
            "modeling",
            "design_studio",
            "surrogate",
            "solution_space",
            "optimization",
            "sensitivity",
        ],
        "design_studio_graph": "design_studio.cad",
        "design_studio_results": "design_studio.cad.results.h5",
    }
    return atomic_json_dump(manifest, folder / "pylcss_project.json")


def validate_project_folder(folder_path: PathLike) -> None:
    """Read all persisted component headers before the live UI is mutated."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise ProjectValidationError(f"Project folder does not exist: {folder}")

    manifest_path = folder / "pylcss_project.json"
    systems_path = folder / "systems.json"
    manifest = None
    if manifest_path.is_file():
        manifest = load_json_object(
            manifest_path,
            required_keys=("format", "version", "components"),
        )
        if manifest.get("format") != "pylcss-project":
            raise ProjectValidationError(
                "pylcss_project.json is not a PyLCSS project manifest."
            )
        _manifest_version(manifest.get("version"))
        components = manifest.get("components")
        if not isinstance(components, list) or not all(
            isinstance(component, str) and component for component in components
        ):
            raise ProjectValidationError(
                "pylcss_project.json field 'components' must be a list of names."
            )
    elif not systems_path.is_file():
        raise ProjectValidationError(
            "The selected folder contains neither pylcss_project.json nor "
            "the systems.json marker used by legacy PyLCSS projects."
        )

    if not systems_path.is_file():
        raise ProjectValidationError(
            "The project is incomplete: systems.json is missing."
        )
    systems = load_json_object(systems_path, required_keys=("systems",))
    if not isinstance(systems.get("systems"), list):
        raise ProjectValidationError("systems.json field 'systems' must be a list.")

    if manifest is None or _manifest_version(manifest.get("version")) < 2:
        return

    for filename, required_keys in _JSON_COMPONENTS.items():
        path = folder / filename
        if not path.is_file():
            raise ProjectValidationError(
                f"The project is incomplete: {filename} is missing."
            )
        load_json_object(path, required_keys=required_keys)

    _validate_result_files(folder)


def _manifest_version(value: object) -> int:
    if isinstance(value, bool):
        raise ProjectValidationError("Project format version must be an integer.")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ProjectValidationError(
            "Project format version must be an integer."
        ) from exc
    if version < 1 or version > _CURRENT_FORMAT_VERSION:
        raise ProjectValidationError(
            f"Project format version {version} is not supported by this build."
        )
    return version


def _validate_result_files(folder: Path) -> None:
    import h5py

    solution_path = folder / "solution_space.h5"
    if not solution_path.is_file():
        raise ProjectValidationError(
            "The project is incomplete: solution_space.h5 is missing."
        )
    with h5py.File(solution_path, "r") as handle:
        if _text_attribute(handle.attrs.get("format")) != "pylcss-solution-space":
            raise ProjectValidationError(
                "solution_space.h5 has an unrecognized format."
            )
        if _integer_attribute(handle.attrs.get("version")) != 1:
            raise ProjectValidationError(
                "solution_space.h5 uses an unsupported version."
            )

    result_path = folder / "design_studio.cad.results.h5"
    if not result_path.is_file():
        raise ProjectValidationError(
            "The project is incomplete: Design Studio results are missing."
        )

    from pylcss.design_studio.result_store import (
        FORMAT_NAME as RESULT_FORMAT,
        FORMAT_VERSION as RESULT_VERSION,
    )

    with h5py.File(result_path, "r") as handle:
        if _text_attribute(handle.attrs.get("format")) != RESULT_FORMAT:
            raise ProjectValidationError(
                "design_studio.cad.results.h5 has an unrecognized format."
            )
        if _integer_attribute(handle.attrs.get("version")) != RESULT_VERSION:
            raise ProjectValidationError(
                "design_studio.cad.results.h5 uses an unsupported version."
            )


def _text_attribute(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _integer_attribute(value: object) -> int:
    if isinstance(value, bool):
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
