# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Download and install the external solver components PyLCSS can launch.

CalculiX (``ccx``), OpenRadioss (``starter_*`` / ``engine_*`` / ``anim_to_vtk``) are external solver
components, not Python packages that ship in ``requirements.txt``.  This script
fetches the upstream release archives,
unpacks them under ``<repo>/external_solvers/<solver>``, and writes the
matching environment variables to ``<repo>/external_solvers/env.txt``.

Usage
-----
    python scripts/install_solvers.py                # all solvers
    python scripts/install_solvers.py --only ccx     # CalculiX only
    python scripts/install_solvers.py --only radioss # OpenRadioss only
    python scripts/install_solvers.py --list         # show what would be installed

Once the components are unpacked, point PyLCSS at them either by sourcing the
written env file or by setting these variables yourself:

    PYLCSS_CALCULIX_CCX             -> full path to ``ccx`` / ``ccx.exe``
    PYLCSS_OPENRADIOSS_STARTER      -> full path to ``starter_*``
    PYLCSS_OPENRADIOSS_ENGINE       -> full path to ``engine_*``
    PYLCSS_OPENRADIOSS_ANIM2VTK     -> full path to ``anim_to_vtk``

The script is intentionally conservative: it never overwrites an existing
install unless ``--force`` is passed, and it verifies SHA-256 hashes when one
is supplied for that platform.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_ROOT = REPO_ROOT / "external_solvers"
ENV_FILE = INSTALL_ROOT / "env.txt"
# PyLCSS's solver backends read this JSON file on startup, so the user does
# not have to source env.txt manually.  Keep the path in sync with
# pylcss.solver_backends.common._load_solver_paths_config.
CONFIG_FILE = INSTALL_ROOT / "solver_paths.json"


@dataclass
class SolverAsset:
    name: str
    url: str
    archive: str
    binary_glob: str  # glob pattern, relative to extraction root
    env_var: str
    sha256: Optional[str] = None
    extra_globs: Dict[str, str] = field(default_factory=dict)  # env_var -> glob


# NOTE: upstream URLs change with each release. The defaults below point at
# release pages that are stable as of 2026-01; pass --url-override to update
# them without editing the script.
SOLVERS: Dict[str, Dict[str, SolverAsset]] = {
    "ccx": {
        # dhondt.de hosts pre-built Windows binaries for the current CalculiX
        # release; only the latest filename is kept on the server, so update
        # the version here whenever a new release lands.
        "Windows": SolverAsset(
            name="CalculiX",
            url="https://www.dhondt.de/calculix_2.23_4win.zip",
            archive="calculix_2.23_4win.zip",
            # IMPORTANT: pick ccx_static.exe.  The same zip also ships
            # ccx_dynamic.exe, but per its bundled README that build requires
            # Intel MKL's `mkl_rt.2.dll` at runtime, which is not packaged.
            # ccx_static.exe is fully self-contained (SPOOLES + PaStiX linked
            # statically) and works out of the box.
            binary_glob="**/ccx_static.exe",
            env_var="PYLCSS_CALCULIX_CCX",
        ),
        # dhondt.de does not ship Linux/macOS binaries; the .tar.bz2 there is
        # source-only.  The most reliable cross-distro path is the OS package
        # manager.  We surface that via the URL handler in install_solver().
        "Linux": SolverAsset(
            name="CalculiX",
            url="apt:calculix-ccx",
            archive="",
            binary_glob="ccx",
            env_var="PYLCSS_CALCULIX_CCX",
        ),
        "Darwin": SolverAsset(
            name="CalculiX",
            url="brew:calculix-ccx",
            archive="",
            binary_glob="ccx",
            env_var="PYLCSS_CALCULIX_CCX",
        ),
    },
    "radioss": {
        # OpenRadioss publishes per-platform zips under a rolling "latest-*"
        # tag.  The /releases/latest/download/ path is a stable GitHub redirect
        # to whichever asset name we request.
        "Windows": SolverAsset(
            name="OpenRadioss",
            url=(
                "https://github.com/OpenRadioss/OpenRadioss/releases/latest/"
                "download/OpenRadioss_win64.zip"
            ),
            archive="OpenRadioss_win64.zip",
            binary_glob="**/starter_win64.exe",
            env_var="PYLCSS_OPENRADIOSS_STARTER",
            extra_globs={
                "PYLCSS_OPENRADIOSS_ENGINE": "**/engine_win64.exe",
                "PYLCSS_OPENRADIOSS_ANIM2VTK": "**/anim_to_vtk*.exe",
            },
        ),
        "Linux": SolverAsset(
            name="OpenRadioss",
            url=(
                "https://github.com/OpenRadioss/OpenRadioss/releases/latest/"
                "download/OpenRadioss_linux64.zip"
            ),
            archive="OpenRadioss_linux64.zip",
            binary_glob="**/starter_linux64_gf",
            env_var="PYLCSS_OPENRADIOSS_STARTER",
            extra_globs={
                "PYLCSS_OPENRADIOSS_ENGINE": "**/engine_linux64_gf",
                "PYLCSS_OPENRADIOSS_ANIM2VTK": "**/anim_to_vtk*",
            },
        ),
        "Darwin": SolverAsset(
            name="OpenRadioss",
            # No official macOS binary is shipped upstream — Apple Silicon
            # builds must come from source.
            url="manual:see https://github.com/OpenRadioss/OpenRadioss#building-from-source",
            archive="",
            binary_glob="starter_*",
            env_var="PYLCSS_OPENRADIOSS_STARTER",
        ),
    }
}


def _platform_key() -> str:
    sysname = platform.system()
    if sysname not in ("Windows", "Linux", "Darwin"):
        raise SystemExit(f"Unsupported platform: {sysname}")
    return sysname


def _download(url: str, target: Path, expected_sha256: Optional[str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> downloading {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PyLCSS-install-solvers/1.0 (+https://github.com/)"},
    )
    try:
        with urllib.request.urlopen(request) as resp, open(target, "wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"\nDownload failed: HTTP {exc.code} for {url}\n"
            "The upstream filename may have changed.  Visit the project page "
            "(dhondt.de for CalculiX, github.com/OpenRadioss/OpenRadioss/"
            "releases for OpenRadioss), grab the correct archive name, and "
            "rerun with --url-override <full-url>."
        ) from exc
    if expected_sha256:
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise SystemExit(
                f"SHA-256 mismatch for {target.name}: expected {expected_sha256}, got {actual}"
            )


def _extract(archive: Path, dest: Path) -> None:
    print(f"  -> unpacking {archive.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.name.endswith(".tgz"):
        mode = "r:gz"
    elif archive.suffixes[-2:] == [".tar", ".bz2"]:
        mode = "r:bz2"
    elif archive.suffix == ".tar":
        mode = "r"
    else:
        raise SystemExit(f"Unknown archive type: {archive.name}")
    with tarfile.open(archive, mode) as tf:
        tf.extractall(dest)


def _find_first(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _print_manual_instructions(asset: SolverAsset) -> None:
    scheme, _, payload = asset.url.partition(":")
    print(f"[{asset.name}] cannot be auto-installed on this platform.")
    if scheme == "apt":
        print(f"  Install via your package manager, e.g.  sudo apt install {payload}")
        print("  Then ensure the resulting binary is on PATH (PyLCSS auto-discovers ccx).")
    elif scheme == "brew":
        print(f"  Install via Homebrew:  brew install {payload}")
        print("  (You may need to tap a third-party formula; CalculiX is not in core homebrew.)")
    elif scheme == "manual":
        print(f"  {payload}")
    print(f"  Then export {asset.env_var}=/full/path/to/component  for PyLCSS to pick it up.")


def _ensure_executable(path: Path) -> None:
    if platform.system() == "Windows":
        return
    try:
        path.chmod(path.stat().st_mode | 0o755)
    except OSError:
        pass


def install_solver(key: str, force: bool, url_override: Optional[str]) -> Dict[str, str]:
    plat = _platform_key()
    plat_assets = SOLVERS.get(key, {})
    asset = plat_assets.get(plat)
    if asset is None:
        raise SystemExit(f"Solver '{key}' has no asset for platform '{plat}'.")

    if url_override:
        asset.url = url_override

    # Non-URL "schemes" describe install paths that this script cannot perform
    # automatically — we surface clear instructions instead of pretending.
    if asset.url.startswith(("apt:", "brew:", "manual:")):
        _print_manual_instructions(asset)
        return {}

    install_dir = INSTALL_ROOT / key
    extract_dir = install_dir / "unpacked"
    archive_path = install_dir / asset.archive

    if extract_dir.exists() and not force:
        print(f"[{asset.name}] already installed at {extract_dir} (use --force to redo).")
    else:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        if archive_path.exists() and not force:
            print(f"[{asset.name}] reusing cached archive {archive_path.name}")
        else:
            _download(asset.url, archive_path, asset.sha256)
        _extract(archive_path, extract_dir)

    resolved: Dict[str, str] = {}
    main = _find_first(extract_dir, asset.binary_glob)
    if main is None:
        raise SystemExit(
            f"[{asset.name}] could not locate '{asset.binary_glob}' under {extract_dir}.  "
            "The release layout may have changed; rerun with --url-override or fix the script."
        )
    _ensure_executable(main)
    resolved[asset.env_var] = str(main)

    for env_var, pattern in asset.extra_globs.items():
        candidate = _find_first(extract_dir, pattern)
        if candidate:
            _ensure_executable(candidate)
            resolved[env_var] = str(candidate)
        else:
            print(f"  ! no match for {pattern}; {env_var} not set")

    return resolved


def write_env_file(env: Dict[str, str]) -> None:
    import json

    INSTALL_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) JSON config that PyLCSS reads automatically on startup.
    if CONFIG_FILE.exists():
        try:
            existing = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged = {**existing, **env}
            else:
                merged = dict(env)
        except Exception:
            merged = dict(env)
    else:
        merged = dict(env)
    CONFIG_FILE.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {CONFIG_FILE}")
    print("  PyLCSS will pick these paths up automatically the next time you run it.")

    # 2) Shell env.txt for users who prefer to source it.
    lines: List[str] = ["# Auto-generated by scripts/install_solvers.py"]
    for key, value in env.items():
        lines.append(f"export {key}={value}")
        lines.append(f"set {key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {ENV_FILE}  (optional - only needed if you launch CCX/Radioss outside PyLCSS).")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(SOLVERS.keys()), help="Install one solver instead of all.")
    parser.add_argument("--force", action="store_true", help="Reinstall even if already extracted.")
    parser.add_argument("--list", action="store_true", help="Show the URLs that would be downloaded.")
    parser.add_argument(
        "--url-override",
        help="Replace the upstream URL for the selected solver (use with --only).",
    )
    args = parser.parse_args(argv)

    plat = _platform_key()
    keys = [args.only] if args.only else list(SOLVERS.keys())

    if args.list:
        for key in keys:
            asset = SOLVERS[key].get(plat)
            if not asset:
                print(f"{key} [{plat}] -> (no asset)")
            elif asset.url.startswith(("apt:", "brew:", "manual:")):
                print(f"{key} [{plat}] -> manual: {asset.url}")
            else:
                print(f"{key} [{plat}] -> {asset.url}")
        return 0

    env: Dict[str, str] = {}
    for key in keys:
        print(f"\n=== Installing {key} ===")
        env.update(install_solver(key, force=args.force, url_override=args.url_override))

    if env:
        write_env_file(env)
    else:
        print("Nothing installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
