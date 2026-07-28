# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Private helpers for replacing files without exposing partial writes."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, Iterator

PathLike = str | os.PathLike[str]


def _sync_file(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers via os.fsync.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_output_path(path: PathLike) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace the target on success."""
    target = Path(path).expanduser()
    if not target.name:
        raise ValueError("An output file path is required.")
    if target.exists() and not target.is_file():
        raise IsADirectoryError(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        yield temporary
        if not temporary.is_file():
            raise FileNotFoundError(
                f"The writer did not create the temporary file for {target}."
            )
        if target.exists():
            shutil.copymode(target, temporary)
        _sync_file(temporary)
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        with suppress(OSError):
            temporary.unlink()


@contextmanager
def atomic_text_writer(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Iterator[IO[str]]:
    """Open a text stream that replaces its destination only after a clean close."""
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding=encoding, newline=newline) as handle:
            yield handle
