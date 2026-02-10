# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""
Full project save/load manager for pylcss.
Handles .pylcss project archives (ZIP-based) containing:
    - System model graphs
    - CAD sessions
    - Optimization settings & results
    - Sensitivity analysis data
    - Solution space data (HDF5)
    - Surrogate models
    - Configuration
"""

import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_VERSION = "2.0.0"
PROJECT_EXTENSION = ".pylcss"


class ProjectManager:
    """
    Full project archive manager.
    
    Project structure (.pylcss is a ZIP archive):
        manifest.json           - version, metadata
        systems/                - system model graphs (.json)
        cad/                    - CAD sessions (.cad)
        optimization/           - optimization_setup.json
        sensitivity/            - sensitivity.json
        solution_space/         - solution_space.json, data.h5
        surrogate/              - surrogate_settings.json, models/*.pkl
        config/                 - settings.json
    """

    @staticmethod
    def save_project(
        filepath: str,
        project_data: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Save complete project to .pylcss archive.
        
        Args:
            filepath: path ending in .pylcss
            project_data: dict with keys matching project sections
            metadata: optional project metadata
        """
        if not filepath.endswith(PROJECT_EXTENSION):
            filepath += PROJECT_EXTENSION

        # Create temp directory for staging
        with tempfile.TemporaryDirectory(prefix="pylcss_save_") as tmpdir:
            # Write manifest
            manifest = {
                "version": PROJECT_VERSION,
                "created": datetime.now().isoformat(),
                "pylcss_version": "1.3.0",
                "metadata": metadata or {},
                "sections": list(project_data.keys()),
            }
            _write_json(os.path.join(tmpdir, "manifest.json"), manifest)

            # Write each section
            for section, data in project_data.items():
                section_dir = os.path.join(tmpdir, section)
                os.makedirs(section_dir, exist_ok=True)

                if section == "systems":
                    _write_json(os.path.join(section_dir, "graphs.json"), data)
                elif section == "cad":
                    _write_json(os.path.join(section_dir, "session.json"), data)
                elif section == "optimization":
                    _write_json(os.path.join(section_dir, "setup.json"), data)
                elif section == "sensitivity":
                    _write_json(os.path.join(section_dir, "analysis.json"), data)
                elif section == "solution_space":
                    # Separate large arrays into HDF5
                    arrays = {}
                    json_data = {}
                    for k, v in data.items():
                        if isinstance(v, (list,)) and len(v) > 1000:
                            arrays[k] = v
                        else:
                            json_data[k] = v
                    _write_json(os.path.join(section_dir, "config.json"), json_data)
                    if arrays:
                        _write_arrays_hdf5(os.path.join(section_dir, "data.h5"), arrays)
                elif section == "surrogate":
                    settings = {k: v for k, v in data.items() if k != "models"}
                    _write_json(os.path.join(section_dir, "settings.json"), settings)
                    # Copy model files
                    if "models" in data:
                        models_dir = os.path.join(section_dir, "models")
                        os.makedirs(models_dir, exist_ok=True)
                        for name, model_path in data["models"].items():
                            if os.path.isfile(model_path):
                                dest = os.path.join(models_dir, f"{name}.pkl")
                                shutil.copy2(model_path, dest)
                elif section == "config":
                    _write_json(os.path.join(section_dir, "settings.json"), data)
                else:
                    _write_json(os.path.join(section_dir, "data.json"), data)

            # Create ZIP archive
            _zip_directory(tmpdir, filepath)

        logger.info(f"Project saved: {filepath}")

    @staticmethod
    def load_project(filepath: str) -> Dict[str, Any]:
        """
        Load complete project from .pylcss archive.
        
        Returns dict with section data.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Project file not found: {filepath}")

        project_data = {}

        with tempfile.TemporaryDirectory(prefix="pylcss_load_") as tmpdir:
            # Extract archive
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(tmpdir)

            # Read manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            if os.path.isfile(manifest_path):
                manifest = _read_json(manifest_path)
                project_data["_manifest"] = manifest
                logger.info(
                    f"Loading project v{manifest.get('version', '?')}, "
                    f"created {manifest.get('created', '?')}"
                )
            else:
                logger.warning("No manifest found in project archive")

            # Read each section
            for item in os.listdir(tmpdir):
                item_path = os.path.join(tmpdir, item)
                if not os.path.isdir(item_path) or item.startswith("_"):
                    continue

                section = item
                section_data = {}

                for file in os.listdir(item_path):
                    file_path = os.path.join(item_path, file)
                    if file.endswith(".json"):
                        section_data.update(_read_json(file_path))
                    elif file.endswith(".h5"):
                        section_data.update(_read_arrays_hdf5(file_path))
                    elif file == "models" and os.path.isdir(file_path):
                        models = {}
                        for model_file in os.listdir(file_path):
                            if model_file.endswith(".pkl"):
                                name = Path(model_file).stem
                                # Copy to persistent location
                                models[name] = os.path.join(file_path, model_file)
                        section_data["models"] = models

                project_data[section] = section_data

        logger.info(f"Project loaded: {filepath}")
        return project_data

    @staticmethod
    def get_project_info(filepath: str) -> Dict:
        """Get project metadata without full load."""
        with zipfile.ZipFile(filepath, "r") as zf:
            if "manifest.json" in zf.namelist():
                with zf.open("manifest.json") as f:
                    return json.load(f)
        return {}

    @staticmethod
    def export_section(
        filepath: str, section: str, data: Any
    ) -> None:
        """Export a single project section to standalone file."""
        ext = Path(filepath).suffix.lower()
        if ext == ".json":
            _write_json(filepath, data)
        elif ext == ".h5":
            _write_arrays_hdf5(filepath, data)
        elif ext == ".csv":
            from pylcss.io_manager.data_io import DataExporter
            import numpy as np

            if isinstance(data, dict) and "data" in data:
                DataExporter.to_csv(filepath, np.array(data["data"]), data.get("columns"))
            else:
                DataExporter.to_csv(filepath, np.array(data))
        else:
            _write_json(filepath, data)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _write_json(filepath: str, data: Any) -> None:
    """Write JSON with numpy support."""
    import numpy as np

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
            if isinstance(obj, (complex, np.complexfloating)):
                return {"__complex__": True, "real": obj.real, "imag": obj.imag}
            return super().default(obj)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def _read_json(filepath: str) -> Dict:
    """Read JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_arrays_hdf5(filepath: str, data: Dict) -> None:
    """Write arrays to HDF5."""
    try:
        import h5py
        import numpy as np

        with h5py.File(filepath, "w") as f:
            for key, value in data.items():
                arr = np.asarray(value)
                f.create_dataset(key, data=arr, compression="gzip")
    except ImportError:
        logger.warning("h5py not available, skipping HDF5 data")


def _read_arrays_hdf5(filepath: str) -> Dict:
    """Read arrays from HDF5."""
    try:
        import h5py
        import numpy as np

        result = {}
        with h5py.File(filepath, "r") as f:
            for key in f:
                result[key] = np.array(f[key])
        return result
    except ImportError:
        logger.warning("h5py not available, skipping HDF5 data")
        return {}


def _zip_directory(source_dir: str, zip_filepath: str) -> None:
    """Create ZIP archive from directory."""
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_name = os.path.relpath(abs_path, source_dir)
                zf.write(abs_path, arc_name)
