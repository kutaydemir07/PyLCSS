"""Private CAD jobs launched with explicit windowless subprocess options.

The job and result files are private, short-lived files written by this process.
They retain numpy arrays and typed manufacturing options without conversion.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
from time import monotonic, sleep

from pylcss.process_utils import headless_subprocess_kwargs


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


def run_cad_job(operation: str, *arguments, timeout_seconds: float = 900,
                progress_callback=None, cancel_callback=None):
    if operation not in {"brep", "step", "step_shape", "mesh", "worker_environment"}:
        raise ValueError(f"Unsupported CAD background operation: {operation}")
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
                                       **headless_subprocess_kwargs())
            started = monotonic()
            previous_progress = ""
            try:
                while True:
                    if cancel_callback is not None and cancel_callback():
                        raise CadJobCancelled("CAD reconstruction cancelled")
                    if progress_callback is not None and progress.is_file():
                        current = progress.read_text(encoding="utf-8")
                        if current != previous_progress:
                            payload, step, total = json.loads(current)
                            progress_callback(payload, None, step, total)
                            previous_progress = current
                    remaining = timeout_seconds - (monotonic() - started)
                    if remaining <= 0:
                        raise RuntimeError(f"CAD {operation} job exceeded {timeout_seconds:g} seconds; "
                                           "the destination file was not replaced.")
                    if process.poll() is not None:
                        break
                    try:
                        process.wait(timeout=min(.1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    _terminate_cad_process(process)
        if process.returncode or not result.is_file():
            error = stderr.read_text(encoding="utf-8", errors="replace")[-4000:]
            output = stdout.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"CAD background job failed: {error or output}")
        with result.open("rb") as stream:
            return pickle.load(stream)


def reconstruct_cad_in_background(payload, *, reconstruction_strategy="auto",
                                  progress_callback=None, cancel_callback=None, **options):
    """Return the native body, report and display mesh from one isolated build."""
    import cadquery as cq
    # These are viewer/analysis caches, never inputs to reconstruction. In
    # particular, transferring an old CAD body would serialize OCC geometry
    # in the parent for no benefit.
    payload = {key: value for key, value in payload.items()
               if key not in {"shape", "cad_shape", "cad_preview_mesh", "design_domain", "validation"}}
    with _cad_job_directory("pylcss_cad_preview_") as directory:
        path = Path(directory) / "preview.brep"
        report = run_cad_job("brep", payload, str(path), reconstruction_strategy, options,
                             progress_callback=progress_callback, cancel_callback=cancel_callback)
        preview_mesh = report.pop("_preview_mesh", None)
        # Only restore the native topology here. Fitting, validation and
        # tessellation all happened in the child, so the GUI never repeats them.
        shape = cq.importers.importBrep(str(path))
    return shape, report, preview_mesh


def main():
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
            _external_write_cad_mesh, _write_step_shape,
        )
        functions = {"brep": _external_write_cad_brep, "step": _external_write_cad_step,
                     "step_shape": _write_step_shape, "mesh": _external_write_cad_mesh}
        if operation not in functions:
            raise ValueError(f"Unsupported CAD background operation: {operation}")
        if operation == "brep":
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
    main()
