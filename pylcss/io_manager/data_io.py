# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
Data import/export for optimization, sensitivity, and surrogate workflows.
Supports: CSV, JSON, HDF5, MAT (MATLAB), Excel, Pickle
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class DataImporter:
    """Import data from various formats."""

    @staticmethod
    def from_csv(
        filepath: str,
        delimiter: str = ",",
        header: bool = True,
        skip_rows: int = 0,
    ) -> Dict:
        """
        Import CSV file.
        
        Returns dict with 'data' (2D array), 'columns' (list), 'n_rows', 'n_cols'.
        """
        import pandas as pd

        df = pd.read_csv(
            filepath,
            delimiter=delimiter,
            header=0 if header else None,
            skiprows=skip_rows,
        )
        columns = list(df.columns)
        data = df.values

        logger.info(f"Imported CSV: {data.shape[0]} rows x {data.shape[1]} cols from {filepath}")
        return {
            "data": data,
            "columns": columns,
            "n_rows": data.shape[0],
            "n_cols": data.shape[1],
            "dtypes": {col: str(df[col].dtype) for col in columns},
        }

    @staticmethod
    def from_json(filepath: str) -> Dict:
        """Import JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Imported JSON: {filepath}")
        return data

    @staticmethod
    def from_hdf5(filepath: str, dataset: Optional[str] = None) -> Dict:
        """
        Import from HDF5 file.
        
        If dataset is None, returns all datasets as a dict.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 import")

        result = {}
        with h5py.File(filepath, "r") as f:
            if dataset:
                result["data"] = np.array(f[dataset])
                result["attrs"] = dict(f[dataset].attrs)
            else:
                def _read_group(group, prefix=""):
                    for key in group:
                        path = f"{prefix}/{key}" if prefix else key
                        if isinstance(group[key], h5py.Dataset):
                            result[path] = {
                                "data": np.array(group[key]),
                                "attrs": dict(group[key].attrs),
                            }
                        elif isinstance(group[key], h5py.Group):
                            _read_group(group[key], path)

                _read_group(f)
                result["_file_attrs"] = dict(f.attrs)

        logger.info(f"Imported HDF5: {filepath}")
        return result

    @staticmethod
    def from_mat(filepath: str) -> Dict:
        """Import MATLAB .mat file."""
        try:
            from scipy.io import loadmat
            data = loadmat(filepath, squeeze_me=True)
            # Remove MATLAB metadata keys
            data = {k: v for k, v in data.items() if not k.startswith("__")}
            logger.info(f"Imported MAT: {filepath}")
            return data
        except ImportError:
            raise ImportError("scipy required for MAT import")

    @staticmethod
    def from_excel(
        filepath: str,
        sheet_name: Optional[str] = None,
    ) -> Dict:
        """Import Excel file (.xlsx, .xls)."""
        try:
            import pandas as pd
            if sheet_name:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                return {
                    "data": df.values,
                    "columns": list(df.columns),
                    "n_rows": len(df),
                }
            else:
                dfs = pd.read_excel(filepath, sheet_name=None)
                result = {}
                for name, df in dfs.items():
                    result[name] = {
                        "data": df.values,
                        "columns": list(df.columns),
                        "n_rows": len(df),
                    }
                logger.info(f"Imported Excel: {list(result.keys())} sheets from {filepath}")
                return result
        except ImportError:
            raise ImportError("pandas + openpyxl required for Excel import")

    @staticmethod
    def from_pickle(filepath: str) -> Any:
        """Import from pickle (for surrogate models, etc.)."""
        import joblib
        data = joblib.load(filepath)
        logger.info(f"Imported pickle: {filepath}")
        return data


class DataExporter:
    """Export data to various formats."""

    @staticmethod
    def to_csv(
        filepath: str,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
        delimiter: str = ",",
        index: bool = False,
    ) -> None:
        """Export data to CSV."""
        import pandas as pd

        if columns:
            df = pd.DataFrame(data, columns=columns)
        else:
            df = pd.DataFrame(data)

        df.to_csv(filepath, sep=delimiter, index=index)
        logger.info(f"Exported CSV: {data.shape[0]} rows to {filepath}")

    @staticmethod
    def to_json(filepath: str, data: Any, indent: int = 2) -> None:
        """Export to JSON with numpy serialization support."""

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, cls=NumpyEncoder)
        logger.info(f"Exported JSON: {filepath}")

    @staticmethod
    def to_hdf5(
        filepath: str,
        datasets: Dict[str, np.ndarray],
        attrs: Optional[Dict] = None,
        compression: str = "gzip",
    ) -> None:
        """Export to HDF5 with compression."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 export")

        with h5py.File(filepath, "w") as f:
            if attrs:
                for k, v in attrs.items():
                    f.attrs[k] = v
            for name, data in datasets.items():
                data = np.asarray(data)
                f.create_dataset(name, data=data, compression=compression)

        logger.info(f"Exported HDF5: {len(datasets)} datasets to {filepath}")

    @staticmethod
    def to_mat(filepath: str, data: Dict) -> None:
        """Export to MATLAB .mat file."""
        from scipy.io import savemat
        # Ensure all values are numpy arrays
        clean_data = {}
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                clean_data[k] = np.array(v)
            else:
                clean_data[k] = v
        savemat(filepath, clean_data)
        logger.info(f"Exported MAT: {filepath}")

    @staticmethod
    def to_excel(
        filepath: str,
        sheets: Dict[str, np.ndarray],
        columns: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Export to Excel with multiple sheets."""
        import pandas as pd

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for name, data in sheets.items():
                cols = columns.get(name) if columns else None
                df = pd.DataFrame(data, columns=cols)
                df.to_excel(writer, sheet_name=name, index=False)

        logger.info(f"Exported Excel: {list(sheets.keys())} sheets to {filepath}")

    @staticmethod
    def to_pickle(filepath: str, data: Any, compress: int = 3) -> None:
        """Export to pickle (for surrogate models, etc.)."""
        import joblib
        joblib.dump(data, filepath, compress=compress)
        logger.info(f"Exported pickle: {filepath}")

    @staticmethod
    def results_to_report(
        filepath: str,
        title: str,
        sections: List[Dict],
        format: str = "html",
    ) -> None:
        """
        Export results as formatted report.
        
        Args:
            filepath: output path
            title: report title
            sections: list of dicts with 'heading', 'text', 'table', 'image' keys
            format: 'html' or 'md'
        """
        if format == "html":
            DataExporter._write_html_report(filepath, title, sections)
        elif format == "md":
            DataExporter._write_md_report(filepath, title, sections)
        else:
            raise ValueError(f"Unknown report format: {format}")

    @staticmethod
    def _write_html_report(filepath: str, title: str, sections: List[Dict]) -> None:
        """Generate HTML report."""
        html = [
            "<!DOCTYPE html><html><head>",
            f"<title>{title}</title>",
            "<style>",
            "body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #1e1f22; color: #e0e0e0; }",
            "h1 { color: #d29922; border-bottom: 2px solid #d29922; padding-bottom: 10px; }",
            "h2 { color: #d29922; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #444; padding: 8px; text-align: left; }",
            "th { background: #2b2d30; color: #d29922; }",
            "tr:nth-child(even) { background: #2b2d30; }",
            ".metric { display: inline-block; margin: 10px; padding: 15px; background: #2b2d30;",
            "  border-radius: 8px; border-left: 4px solid #d29922; }",
            ".metric-value { font-size: 24px; font-weight: bold; color: #d29922; }",
            ".metric-label { font-size: 12px; color: #888; }",
            "img { max-width: 100%; border-radius: 4px; }",
            "</style></head><body>",
            f"<h1>{title}</h1>",
            f"<p>Generated by pylcss</p>",
        ]

        for section in sections:
            if "heading" in section:
                html.append(f"<h2>{section['heading']}</h2>")
            if "text" in section:
                html.append(f"<p>{section['text']}</p>")
            if "metrics" in section:
                for name, value in section["metrics"].items():
                    html.append(
                        f'<div class="metric"><div class="metric-value">{value}</div>'
                        f'<div class="metric-label">{name}</div></div>'
                    )
            if "table" in section:
                table = section["table"]
                html.append("<table>")
                if "headers" in table:
                    html.append("<tr>" + "".join(f"<th>{h}</th>" for h in table["headers"]) + "</tr>")
                for row in table.get("rows", []):
                    html.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
                html.append("</table>")
            if "image" in section:
                import base64
                with open(section["image"], "rb") as img_f:
                    img_data = base64.b64encode(img_f.read()).decode()
                ext = Path(section["image"]).suffix.lstrip(".")
                html.append(f'<img src="data:image/{ext};base64,{img_data}" />')

        html.append("</body></html>")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        logger.info(f"Exported HTML report: {filepath}")

    @staticmethod
    def _write_md_report(filepath: str, title: str, sections: List[Dict]) -> None:
        """Generate Markdown report."""
        lines = [f"# {title}", "", f"*Generated by pylcss*", ""]

        for section in sections:
            if "heading" in section:
                lines.append(f"## {section['heading']}")
                lines.append("")
            if "text" in section:
                lines.append(section["text"])
                lines.append("")
            if "metrics" in section:
                for name, value in section["metrics"].items():
                    lines.append(f"- **{name}**: {value}")
                lines.append("")
            if "table" in section:
                table = section["table"]
                if "headers" in table:
                    lines.append("| " + " | ".join(str(h) for h in table["headers"]) + " |")
                    lines.append("| " + " | ".join("---" for _ in table["headers"]) + " |")
                for row in table.get("rows", []):
                    lines.append("| " + " | ".join(str(v) for v in row) + " |")
                lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported Markdown report: {filepath}")
