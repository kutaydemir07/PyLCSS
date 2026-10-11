# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Robust data import and preparation utilities for surrogate modeling."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
    ".npy",
    ".npz",
]


class DataImporter:
    """Load, inspect, and prepare tabular datasets from various formats."""

    @staticmethod
    def load_file(filepath: str | Path) -> pd.DataFrame:
        """Load tabular data from file, auto-detecting format and delimiters."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix in (".csv", ".tsv", ".txt"):
            return DataImporter._load_delimited(path)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif suffix == ".json":
            return DataImporter._load_json(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        elif suffix == ".npy":
            arr = np.load(path)
            if arr.ndim == 1:
                return pd.DataFrame(arr, columns=["y"])
            elif arr.ndim == 2:
                cols = [f"col_{i}" for i in range(arr.shape[1])]
                return pd.DataFrame(arr, columns=cols)
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {arr.ndim}")
        elif suffix == ".npz":
            npz = np.load(path)
            data_dict = {}
            for k in npz.files:
                val = npz[k]
                if val.ndim == 1:
                    data_dict[k] = val
                elif val.ndim == 2:
                    for col_i in range(val.shape[1]):
                        data_dict[f"{k}_{col_i}"] = val[:, col_i]
            if not data_dict:
                raise ValueError("NPZ file contains no array entries.")
            return pd.DataFrame(data_dict)
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

    @staticmethod
    def _load_delimited(path: Path) -> pd.DataFrame:
        """Read CSV/TSV/TXT with delimiter autodetection."""
        # Detect delimiter using sample text
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(8192)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",\t; |")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = None

        if delimiter:
            try:
                return pd.read_csv(path, sep=delimiter)
            except Exception:
                pass

        # Fallback attempts: comma, semicolon, tab, whitespace
        for sep in [",", ";", "\t", r"\s+"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python" if sep == r"\s+" else "c")
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue

        # Final default fallback
        return pd.read_csv(path)

    @staticmethod
    def _load_json(path: Path) -> pd.DataFrame:
        """Load JSON into DataFrame with orient fallback."""
        try:
            return pd.read_json(path)
        except Exception:
            import json

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            elif isinstance(raw, dict):
                # Check for inputs/outputs structure
                if "inputs" in raw and "outputs" in raw:
                    df_in = pd.DataFrame(raw["inputs"])
                    df_out = pd.DataFrame(raw["outputs"])
                    return pd.concat([df_in, df_out], axis=1)
                return pd.DataFrame(raw)
            raise ValueError("Unable to parse JSON format into tabular data.")

    @staticmethod
    def inspect_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Return metadata for each column: name, dtype, null_count, sample_vals."""
        info = []
        for col in df.columns:
            series = df[col]
            # Test if column can be coerced to numeric
            coerced = pd.to_numeric(series, errors="coerce")
            is_numeric = coerced.notna().sum() > 0.5 * len(series)
            null_count = int(series.isna().sum())
            coerced_nulls = int(coerced.isna().sum()) if is_numeric else null_count
            samples = [str(v) for v in series.dropna().head(3).tolist()]
            info.append(
                {
                    "name": str(col),
                    "dtype": str(series.dtype),
                    "is_numeric": is_numeric,
                    "null_count": null_count,
                    "coerced_nulls": coerced_nulls,
                    "sample_values": samples,
                }
            )
        return info

    @staticmethod
    def guess_target_columns(
        df: pd.DataFrame,
        target_name_hint: str | None = None,
    ) -> list[str]:
        """Guess likely output/target columns based on naming or position."""
        cols = [str(c) for c in df.columns]
        if not cols:
            return []

        # If a hint is provided (e.g. from target node)
        if target_name_hint:
            hint_lower = target_name_hint.lower()
            for c in cols:
                if c.lower() == hint_lower:
                    return [c]

        # Common target names
        candidates = ["target", "output", "y", "response", "label", "loss", "stress", "obj"]
        for cand in candidates:
            for c in cols:
                if c.lower() == cand or cand in c.lower():
                    return [c]

        # Default: last numeric column
        numeric_cols = [
            c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
        ]
        if numeric_cols:
            return [numeric_cols[-1]]

        return [cols[-1]]

    @staticmethod
    def clean_and_prepare(
        df: pd.DataFrame,
        input_cols: list[str],
        output_cols: list[str],
        handle_missing: str = "drop",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Convert selected columns to numeric, clean missing values, and split."""
        if not input_cols:
            raise ValueError("At least one input feature column must be selected.")
        if not output_cols:
            raise ValueError("At least one target output column must be selected.")

        # Subset dataframe
        selected_cols = list(input_cols) + list(output_cols)
        working_df = df[selected_cols].copy()

        # Coerce all selected columns to numeric
        for col in selected_cols:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

        total_rows = len(working_df)
        nan_mask = working_df.isna().any(axis=1) | working_df.isin([np.inf, -np.inf]).any(axis=1)
        dropped_rows = int(nan_mask.sum())

        if handle_missing == "drop":
            clean_df = working_df[~nan_mask]
        elif handle_missing in ("impute_mean", "mean"):
            clean_df = working_df.fillna(working_df.mean())
            dropped_rows = 0
        elif handle_missing in ("impute_median", "median"):
            clean_df = working_df.fillna(working_df.median())
            dropped_rows = 0
        else:
            clean_df = working_df[~nan_mask]

        if len(clean_df) < 2:
            raise ValueError(
                f"Dataset contains too few valid samples ({len(clean_df)} remaining) "
                f"after handling missing values. Need at least 2 samples."
            )

        X = clean_df[input_cols].values.astype(np.float64)
        y = clean_df[output_cols].values.astype(np.float64)

        if len(output_cols) == 1:
            y = y.ravel()

        if 0.0 < test_size < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = np.empty((0, X.shape[1])), np.empty((0, *y.shape[1:]))

        stats = {
            "total_raw_rows": total_rows,
            "valid_rows": len(clean_df),
            "dropped_rows": dropped_rows,
            "n_features": len(input_cols),
            "n_targets": len(output_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        return X_train, X_test, y_train, y_test, stats
