"""Private CAD jobs launched with explicit windowless subprocess options.

The job and result files are private, short-lived files written by this process.
They retain numpy arrays and typed manufacturing options without conversion.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
import json
import logging
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import monotonic, sleep
from uuid import uuid4

from pylcss.process_utils import headless_subprocess_kwargs


logger = logging.getLogger(__name__)


class _CadJobPickler(pickle.Pickler):
    """Transfer CAD via text BREP, excluding the viewer's triangulation.

    CadQuery's binary Shape pickle can fail to reimport large fitted bodies.
    Keep geometry outside pickle and deduplicate references in result payloads.
    """
    def __init__(self, stream, directory):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.directory = Path(directory)
        self.shapes = {}

    def persistent_id(self, obj):
        import cadquery as cq
        if not isinstance(obj, (cq.Shape, cq.Workplane)):
            return None
        from OCP.BRepTools import BRepTools
        from OCP.TopTools import TopTools_FormatVersion_VERSION_1

        shape = obj.val() if isinstance(obj, cq.Workplane) else obj
        key = id(shape)
        if key not in self.shapes:
            path = self.directory / f"shape_{len(self.shapes)}.brep"
            if not BRepTools.Write_s(shape.wrapped, str(path), False, False,
                                    TopTools_FormatVersion_VERSION_1):
                raise RuntimeError("Could not transfer the cached CAD body to the export process")
            self.shapes[key] = path.name
        return ("pylcss-cad-brep-v1", self.shapes[key], isinstance(obj, cq.Workplane),
                bool(shape.forConstruction))


class _CadJobUnpickler(pickle.Unpickler):
    def __init__(self, stream, directory):
        super().__init__(stream)
        self.directory = Path(directory)
        self.shapes = {}

    def persistent_load(self, reference):
        tag, name, workplane, construction = reference
        if tag != "pylcss-cad-brep-v1" or Path(name).name != name:
            raise pickle.UnpicklingError("Invalid private CAD body reference")
        import cadquery as cq
        if name not in self.shapes:
            self.shapes[name] = cq.Shape.importBrep(str(self.directory / name))
        shape = self.shapes[name]
        shape.forConstruction = construction
        return cq.Workplane(obj=shape) if workplane else shape


class CadJobCancelled(RuntimeError):
    """An isolated CAD job was stopped before it published a result."""


class CadJobFailed(RuntimeError):
    """A concise user-facing reason, with the complete worker diagnostics."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def _new_failure_checkpoint_directory() -> Path:
    """Reserve a durable diagnostic destination without doing success-path IO."""
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "PyLCSS" / "cad_failures" / f"{stamp}_{uuid4().hex[:12]}"


def _persist_cad_failure_diagnostics(folder, directory, arguments, reason, returncode, *, operation='brep'):
    """Retain portable diagnostics without executable private job pickles."""
    folder, directory = Path(folder), Path(directory)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("stdout.log", "stderr.log", "progress.json", "result.error.json"):
            source = directory / name
            if source.is_file():
                shutil.copyfile(source, folder / name)
        (folder / "failure.json").write_text(json.dumps({
            "operation": operation, "error": reason, "worker_returncode": returncode,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Could not retain CAD worker diagnostics in %s", folder, exc_info=True)

    if operation in {'step', 'step_shape', 'fea', 'lattice', 'lattice_display'}:
        try:
            # Private job inputs disappear at cleanup. Preserve their portable
            # CAD bodies and any rejected STEP transfer for reproduction.
            for source in directory.glob('shape_*.brep'):
                shutil.copyfile(source, folder / ('input_' + source.name))
            if operation in {'step', 'step_shape'}:
                rejected = Path(arguments[1])
                if rejected.is_file() and rejected.stat().st_size:
                    shutil.copyfile(rejected, folder / 'failed_export.step')
        except (OSError, TypeError, IndexError):
            logger.warning('Could not retain failed CAD geometry in %s', folder, exc_info=True)

    # A numerical source mesh makes even failures before the AMRTO checkpoint
    # reproducible, without retaining executable pickle input or OCC caches.
    try:
        payload = arguments[0]
        mesh = payload.get("recovered_shape") if isinstance(payload, dict) else None
        if mesh is None and isinstance(payload, dict) and "vertices" in payload and "faces" in payload:
            mesh = payload
        if isinstance(mesh, dict) and mesh.get("vertices") is not None and mesh.get("faces") is not None:
            import numpy as np
            np.savez_compressed(folder / "source_mesh.npz",
                                vertices=np.asarray(mesh["vertices"], dtype=float),
                                faces=np.asarray(mesh["faces"], dtype=np.int64))
    except (OSError, TypeError, ValueError):
        logger.warning("Could not retain the failed CAD source mesh in %s", folder, exc_info=True)
    try:
        return folder if folder.is_dir() and any(path.is_file() for path in folder.iterdir()) else None
    except OSError:
        return None


@contextmanager
def _cad_job_directory(prefix):
    directory = tempfile.TemporaryDirectory(prefix=prefix)
    try:
        yield directory.name
    finally:
        # Windows interpreter launchers can release inherited log handles just
        # after Popen reports exit. Allow that short delay before removing the
        # private job directory, including after Stop or a timeout.
        for attempt in range(20):
            try:
                directory.cleanup()
                break
            except PermissionError:
                if attempt == 19:
                    raise
                sleep(.05)


def _terminate_cad_process(process):
    if os.name == "nt":
        # Windows venv executables launch another interpreter. Terminating
        # only the launcher leaves native CAD work and its file handles alive.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, **headless_subprocess_kwargs())
    if process.poll() is None:
        process.kill()
    process.wait()


def _cad_job_process_options(operation):
    options = headless_subprocess_kwargs()
    if operation in {"lattice", "lattice_display"}:
        # Native numerical pools otherwise compete for every core while Qt
        # renders the existing model. Apply limits only to the child process.
        limit = max(1, min(4, (os.cpu_count() or 1) - 1))
        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VTK_SMP_MAX_THREADS",
        ):
            try:
                requested = int(environment.get(name, limit))
            except ValueError:
                requested = limit
            environment[name] = str(max(1, min(limit, requested)))
        options["env"] = environment
        if os.name == "nt":
            options["creationflags"] |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return options


def run_cad_job(operation: str, *arguments, timeout_seconds: float | None = None,
                progress_callback=None, cancel_callback=None):
    if operation not in {"brep", "step", "step_shape", "mesh", "surface_layout", "display_mesh", "fea", "lattice", "lattice_display", "worker_environment"}:
        raise ValueError(f"Unsupported CAD background operation: {operation}")
    failure_checkpoint = None
    if operation == "brep":
        # The AMRTO worker writes fitted geometry only when a fit is rejected.
        # Successful builds never create this directory or export stage files.
        payload, destination, *optional = arguments
        strategy = optional[0] if optional else "auto"
        options = dict(optional[1] or {}) if len(optional) > 1 else {}
        failure_checkpoint = Path(options.get("failure_checkpoint_dir") or _new_failure_checkpoint_directory()).resolve()
        options["failure_checkpoint_dir"] = str(failure_checkpoint)
        arguments = (payload, destination, strategy, options)
    elif operation in {'step', 'step_shape', 'fea', 'lattice', 'lattice_display'}:
        failure_checkpoint = _new_failure_checkpoint_directory().resolve()
    with _cad_job_directory("pylcss_cad_job_") as directory:
        job = Path(directory) / "job.pkl"
        result = Path(directory) / "result.pkl"
        progress = Path(directory) / "progress.json"
        with job.open("wb") as stream:
            _CadJobPickler(stream, directory).dump((operation, arguments))
        command = [sys.executable, "-m", __name__, str(job), str(result)]
        # Files prevent verbose OCC output from filling a pipe while we relay
        # progress and check Stop. Waiting releases the parent's Python GIL.
        stdout = Path(directory) / "stdout.log"
        stderr = Path(directory) / "stderr.log"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err,
                                       cwd=Path(__file__).resolve().parents[4],
                                       **_cad_job_process_options(operation))
            started = monotonic()
            previous_progress = ""
            try:
                while True:
                    if cancel_callback is not None and cancel_callback():
                        raise CadJobCancelled("CAD reconstruction cancelled")
                    if progress_callback is not None and progress.is_file():
                        try:
                            current = progress.read_text(encoding="utf-8")
                        except (PermissionError, FileNotFoundError):
                            # Windows can briefly deny opening a status file
                            # during atomic replacement. Keep the job alive.
                            current = previous_progress
                        if current != previous_progress:
                            payload, step, total = json.loads(current)
                            progress_callback(payload, None, step, total)
                            previous_progress = current
                    remaining = None if timeout_seconds is None else timeout_seconds - (monotonic() - started)
                    if remaining is not None and remaining <= 0:
                        raise RuntimeError(f"CAD {operation} job exceeded {timeout_seconds:g} seconds; "
                                           "the destination file was not replaced.")
                    if process.poll() is not None:
                        break
                    try:
                        process.wait(timeout=.1 if remaining is None else min(.1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    _terminate_cad_process(process)
        if process.returncode or not result.is_file():
            details = stderr.read_text(encoding="utf-8", errors="replace")
            if not details.strip():
                details = stdout.read_text(encoding="utf-8", errors="replace")
            failure_path = result.with_suffix(".error.json")
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                reason = failure["message"]
            else:
                reason = next((line.strip() for line in reversed(details.splitlines()) if line.strip()),
                              f"Worker exited with code {process.returncode}")
                if operation in {'fea', 'lattice', 'lattice_display'} and process.returncode:
                    label = {'fea': 'CAD analysis', 'lattice': 'Lattice generation',
                             'lattice_display': 'Lattice display'}[operation]
                    reason = f"{label} worker exited with code {process.returncode}: {reason}"
            if failure_checkpoint is not None:
                saved = _persist_cad_failure_diagnostics(
                    failure_checkpoint, directory, arguments, reason, process.returncode, operation=operation)
                if saved is not None and str(saved) not in reason:
                    reason += f"\nDiagnostics saved to: {saved}"
            logger.error("CAD %s worker diagnostics:\n%s", operation, details)
            label = {'fea': 'CAD analysis', 'lattice': 'Lattice generation',
                     'lattice_display': 'Lattice display'}.get(operation, 'CAD job')
            raise CadJobFailed(f"{label} failed: {reason}", details)
        with result.open("rb") as stream:
            return pickle.load(stream)


def reconstruct_cad_in_background(payload, *, reconstruction_strategy="auto",
                                  progress_callback=None, cancel_callback=None,
                                  preview_only=False, **options):
    """Return the native body, report and display mesh from one isolated build.

    ``preview_only`` is retained for old callers; viewing and export now share
    the same AMRTO construction.
    """
    import cadquery as cq
    # These are viewer/analysis caches, never inputs to reconstruction. In
    # particular, transferring an old CAD body would serialize OCC geometry
    # in the parent for no benefit.
    payload = {key: value for key, value in payload.items()
               if key not in {"shape", "cad_shape", "cad_preview_mesh", "design_domain", "validation"}}
    with _cad_job_directory("pylcss_cad_build_") as directory:
        path = Path(directory) / "cad.brep"
        report = run_cad_job("brep", payload, str(path), reconstruction_strategy, options,
                             progress_callback=progress_callback, cancel_callback=cancel_callback)
        preview_mesh = report.pop("_preview_mesh", None)
        # Only restore the native topology here. Fitting, validation and
        # tessellation all happened in the child, so the GUI never repeats them.
        shape = cq.importers.importBrep(str(path))
    return shape, report, preview_mesh


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    job = Path(sys.argv[1])
    with job.open("rb") as stream:
        operation, arguments = _CadJobUnpickler(stream, job.parent).load()
    if operation == "worker_environment":
        import ctypes
        answer = {"pid": os.getpid(), "executable": sys.executable,
                  "console_window": int(ctypes.windll.kernel32.GetConsoleWindow()) if os.name == "nt" else 0}
    else:
        from .execution_workers import (
            _external_write_cad_brep, _external_write_cad_step,
            _external_write_cad_mesh, _write_step_shape, _external_prepare_cad_display,
        )
        functions = {"brep": _external_write_cad_brep, "step": _external_write_cad_step,
                     "step_shape": _write_step_shape, "mesh": _external_write_cad_mesh,
                     "display_mesh": _external_prepare_cad_display}
        if operation == 'lattice':
            from .execution_workers import _external_generate_lattice
            functions['lattice'] = _external_generate_lattice
        if operation == 'lattice_display':
            from .execution_workers import _external_prepare_lattice_display
            functions['lattice_display'] = _external_prepare_lattice_display
        if operation == 'surface_layout':
            from pylcss.cae.design_studio.topology_optimization.geometry.amrto_surface import prepare_amrto_surface
            functions['surface_layout'] = prepare_amrto_surface
        if operation == 'fea':
            from pylcss.cae.design_studio.topology_optimization.verification.fea_analysis import run_topopt_fea
            def isolated_fea(result, material, supports, loads, options, *, progress_callback):
                def analysis_progress(stage):
                    progress_callback({'stage': stage}, None, 0, 0)
                return run_topopt_fea(result, material, supports, loads,
                                      progress_callback=analysis_progress, **options)
            functions['fea'] = isolated_fea
        if operation not in functions:
            raise ValueError(f"Unsupported CAD background operation: {operation}")
        if operation in {"brep", "surface_layout", "fea", "lattice", "lattice_display"}:
            def progress_callback(payload, _density, step, total):
                path = job.parent / "progress.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps([payload, step, total]), encoding="utf-8")
                try:
                    os.replace(temporary, path)
                except PermissionError:
                    # Windows may briefly deny replacement while the parent
                    # reads the previous status. A later milestone can retry.
                    pass
            answer = functions[operation](*arguments, progress_callback=progress_callback)
        else:
            answer = functions[operation](*arguments)
    with Path(sys.argv[2]).open("wb") as stream:
        pickle.dump(answer, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback
        # Structured reasons travel separately from developer diagnostics.
        Path(sys.argv[2]).with_suffix(".error.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}), encoding="utf-8")
        traceback.print_exc()
        raise SystemExit(1)
