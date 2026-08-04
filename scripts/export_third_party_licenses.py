# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE for details.
"""Export license metadata from the Python environment running this script.

Python wheels may retain notices in ``*.dist-info`` metadata or inside the
installed package itself. This utility copies both forms into one documented
directory so an installed PyLCSS user can inspect and redistribute them
without searching the runtime manually.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable


_NOTICE_PREFIXES = (
    "authors",
    "copying",
    "copyright",
    "licence",
    "license",
    "notice",
)


def _safe_name(value: str) -> str:
    """Return a stable Windows-safe directory name."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def _metadata_summary(value: str | None, limit: int = 500) -> str:
    """Normalize verbose legacy metadata without duplicating whole licenses."""

    normalized = " ".join((value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _is_license_path(path: PurePosixPath) -> bool:
    """Return whether a safe wheel path looks like license material."""

    unsafe_part = any(part in {"", ".."} for part in path.parts)
    if path.is_absolute() or not path.parts or unsafe_part:
        return False
    lower_parts = tuple(part.casefold() for part in path.parts)
    return path.name.casefold().startswith(_NOTICE_PREFIXES) or any(
        part in {"license", "licenses", "licence", "licences"}
        for part in lower_parts
    )


def _license_files(distribution: metadata.Distribution) -> Iterable[PurePosixPath]:
    """Yield all packaged license/notice paths declared by a wheel."""

    for packaged_path in distribution.files or ():
        path = PurePosixPath(str(packaged_path).replace("\\", "/"))
        if _is_license_path(path):
            yield path


def export_licenses(output_directory: Path) -> list[dict[str, object]]:
    """Copy packaged notices and return a serializable distribution index."""

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    distributions = sorted(
        metadata.distributions(),
        key=lambda dist: (dist.metadata.get("Name", "").casefold(), dist.version),
    )
    for distribution in distributions:
        package_name = distribution.metadata.get("Name") or "unknown"
        package_directory = output_directory / _safe_name(
            f"{package_name}-{distribution.version}"
        )
        copied: list[str] = []

        for packaged_path in sorted(set(_license_files(distribution)), key=str):
            source = Path(distribution.locate_file(packaged_path))
            if not source.is_file():
                continue
            # Preserve the complete wheel-relative path. Package-internal
            # trees (for example CasADi's bundled solver licenses) can contain
            # identically named files whose directory context is essential.
            relative = Path(*packaged_path.parts)
            destination = package_directory / relative
            destination.resolve().relative_to(package_directory.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative.as_posix())

        record: dict[str, object] = {
            "name": package_name,
            "version": distribution.version,
            "license_expression": _metadata_summary(
                distribution.metadata.get("License-Expression")
            ),
            "license_metadata": _metadata_summary(distribution.metadata.get("License")),
            "homepage": distribution.metadata.get("Home-page", ""),
            "project_urls": distribution.metadata.get_all("Project-URL") or [],
            "copied_files": copied,
        }
        records.append(record)

    (output_directory / "INDEX.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    index_lines = [
        "PyLCSS installed Python package licenses",
        "",
        "Generated from the wheel records and metadata of the actual isolated",
        "Python runtime. Paths are preserved relative to each distribution.",
        "An empty Files field means that the distribution declared license",
        "metadata but did not include a separate license file in its wheel.",
        "",
    ]
    for record in records:
        license_value = str(
            record["license_expression"] or record["license_metadata"] or "unspecified"
        ).replace("\r", " ").replace("\n", " ")
        files = ", ".join(record["copied_files"]) or "(none packaged)"
        index_lines.extend(
            (
                f"{record['name']} {record['version']}",
                f"  License: {license_value}",
                f"  Files: {files}",
                "",
            )
        )
    (output_directory / "INDEX.txt").write_text(
        "\n".join(index_lines), encoding="utf-8"
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory that will receive INDEX files and copied notices.",
    )
    args = parser.parse_args()
    records = export_licenses(args.output)
    copied_count = sum(len(record["copied_files"]) for record in records)
    print(
        f"Indexed {len(records)} Python distributions and copied "
        f"{copied_count} license/notice files to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
