"""Private CAD jobs launched with explicit windowless subprocess options.

The job and result files are private, short-lived files written by this process.
They retain numpy arrays and typed manufacturing options without conversion.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile

from pylcss.process_utils import headless_subprocess_kwargs


def run_cad_job(operation: str, *arguments):
    if operation not in {"brep", "step", "worker_environment"}:
        raise ValueError(f"Unsupported CAD background operation: {operation}")
    with tempfile.TemporaryDirectory(prefix="pylcss_cad_job_") as directory:
        job = Path(directory) / "job.pkl"
        result = Path(directory) / "result.pkl"
        with job.open("wb") as stream:
            pickle.dump((operation, arguments), stream, protocol=pickle.HIGHEST_PROTOCOL)
        command = [sys.executable, "-m", __name__, str(job), str(result)]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", cwd=Path(__file__).resolve().parents[4],
                                   **headless_subprocess_kwargs())
        if completed.returncode or not result.is_file():
            raise RuntimeError(f"CAD background job failed: {completed.stderr[-4000:] or completed.stdout[-4000:]}")
        with result.open("rb") as stream:
            return pickle.load(stream)


def main():
    with Path(sys.argv[1]).open("rb") as stream:
        operation, arguments = pickle.load(stream)
    if operation == "worker_environment":
        import ctypes
        answer = {"pid": os.getpid(), "executable": sys.executable,
                  "console_window": int(ctypes.windll.kernel32.GetConsoleWindow()) if os.name == "nt" else 0}
    else:
        from .execution_workers import _external_write_cad_brep, _external_write_cad_step
        functions = {"brep": _external_write_cad_brep, "step": _external_write_cad_step}
        if operation not in functions:
            raise ValueError(f"Unsupported CAD background operation: {operation}")
        answer = functions[operation](*arguments)
    with Path(sys.argv[2]).open("wb") as stream:
        pickle.dump(answer, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
