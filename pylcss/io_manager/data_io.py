# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Data export for optimization, sensitivity, and surrogate workflows."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pylcss.io_manager._atomic import PathLike, atomic_output_path
from pylcss.io_manager._reports import ReportFormat, ReportSection, write_report
from pylcss.io_manager.project_io import atomic_json_dump

__all__ = ["DataExporter"]

logger = logging.getLogger(__name__)


def _numpy_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


class DataExporter:
    """Atomically export tabular, scientific, and report data."""

    @staticmethod
    def to_csv(
        filepath: PathLike,
        data: ArrayLike,
        columns: Sequence[str] | None = None,
        delimiter: str = ",",
        index: bool = False,
    ) -> None:
        """Export data to CSV."""
        import pandas as pd

        frame = pd.DataFrame(data, columns=columns)
        with atomic_output_path(filepath) as temporary:
            frame.to_csv(temporary, sep=delimiter, index=index)
        logger.info("Exported CSV with %d rows to %s", len(frame.index), filepath)

    @staticmethod
    def to_json(filepath: PathLike, data: Any, indent: int = 2) -> None:
        """Export standards-compliant JSON with NumPy scalar and array support."""
        atomic_json_dump(
            data,
            filepath,
            indent=indent,
            default=_numpy_json_default,
            allow_nan=False,
        )
        logger.info("Exported JSON to %s", filepath)

    @staticmethod
    def to_hdf5(
        filepath: PathLike,
        datasets: Mapping[str, ArrayLike],
        attrs: Mapping[str, Any] | None = None,
        compression: str | None = "gzip",
    ) -> None:
        """Export datasets and file attributes to HDF5."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 export.") from exc

        with atomic_output_path(filepath) as temporary:
            with h5py.File(temporary, "w") as handle:
                for key, value in (attrs or {}).items():
                    handle.attrs[key] = value
                for name, data in datasets.items():
                    array = np.asarray(data)
                    options = {}
                    if compression is not None and array.ndim > 0 and array.size > 0:
                        options["compression"] = compression
                    handle.create_dataset(name, data=array, **options)

        logger.info("Exported %d HDF5 datasets to %s", len(datasets), filepath)

    @staticmethod
    def to_mat(filepath: PathLike, data: Mapping[str, Any]) -> None:
        """Export values to a .mat file."""
        from scipy.io import savemat

        clean_data = {
            key: np.asarray(value) if isinstance(value, (list, tuple)) else value
            for key, value in data.items()
        }
        with atomic_output_path(filepath) as temporary:
            savemat(temporary, clean_data, appendmat=False)
        logger.info("Exported .mat data to %s", filepath)

    @staticmethod
    def to_excel(
        filepath: PathLike,
        sheets: Mapping[str, ArrayLike],
        columns: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Export one or more worksheets to an Excel workbook."""
        import pandas as pd

        with atomic_output_path(filepath) as temporary:
            with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
                for name, data in sheets.items():
                    sheet_columns = columns.get(name) if columns is not None else None
                    frame = pd.DataFrame(data, columns=sheet_columns)
                    frame.to_excel(writer, sheet_name=name, index=False)

        logger.info("Exported %d Excel sheets to %s", len(sheets), filepath)

    @staticmethod
    def to_pickle(filepath: PathLike, data: Any, compress: int = 3) -> None:
        """Export Python-only data with joblib compression."""
        import joblib

        with atomic_output_path(filepath) as temporary:
            joblib.dump(data, temporary, compress=compress)
        logger.info("Exported joblib data to %s", filepath)

    @staticmethod
    def results_to_report(
        filepath: PathLike,
        title: str,
        sections: Sequence[ReportSection],
        format: ReportFormat = "html",
    ) -> None:
        """Export an escaped HTML or Markdown report."""
        write_report(filepath, title, sections, format)
        logger.info("Exported %s report to %s", format, filepath)
