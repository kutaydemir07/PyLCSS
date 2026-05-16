# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Download and install the external solver components PyLCSS can launch.

CalculiX (``ccx``), OpenRadioss (``starter_*`` / ``engine_*`` / ``anim_to_vtk``)
and FreeCAD (``FreeCAD.exe`` for the interactive sketch/FEM authoring node) are
external components, not Python packages that ship in ``requirements.txt``.
This script fetches the upstream release archives, unpacks them under
``<repo>/external_solvers/<solver>``, and writes the matching environment
variables to ``<repo>/external_solvers/env.txt``.

Usage
-----
    python scripts/install_solvers.py                # everything
    python scripts/install_solvers.py --only ccx     # CalculiX only
    python scripts/install_solvers.py --only radioss # OpenRadioss only
    python scripts/install_solvers.py --only freecad # FreeCAD only
    python scripts/install_solvers.py --list         # show what would be installed

Once the components are unpacked, point PyLCSS at them either by sourcing the
written env file or by setting these variables yourself:

    PYLCSS_CALCULIX_CCX             -> full path to ``ccx`` / ``ccx.exe``
    PYLCSS_OPENRADIOSS_STARTER      -> full path to ``starter_*``
    PYLCSS_OPENRADIOSS_ENGINE       -> full path to ``engine_*``
    PYLCSS_OPENRADIOSS_ANIM2VTK     -> full path to ``anim_to_vtk``
    PYLCSS_FREECAD_EXE              -> full path to ``FreeCAD.exe`` / ``FreeCAD`` / ``FreeCAD.AppImage``

The script is intentionally conservative: it never overwrites an existing
install unless ``--force`` is passed, and it verifies SHA-256 hashes when one
is supplied for that platform.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
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
    },
    "freecad": {
        # FreeCAD bakes the version into every asset name, so the
        # /releases/latest/download/<name> redirect only stays valid while
        # 1.1.1 is the latest tag.  When upstream ships 1.1.2 / 1.2.0 the
        # default URL must be bumped here (or overridden via --url-override).
        #
        # We default to the Windows INSTALLER (.exe) because the portable 7z
        # uses LZMA2 + BCJ2 which py7zr can't decompress, and most users
        # don't have 7-Zip CLI installed.  install_solver() special-cases
        # the .exe suffix below: download -> launch installer wizard ->
        # auto-detect the Program Files install on completion.
        #
        # Power users who DO have 7z.exe on PATH can switch to the portable
        # build with:
        #   python scripts/install_solvers.py --only freecad \
        #     --url-override https://github.com/FreeCAD/FreeCAD/releases/latest/download/FreeCAD_1.1.1-Windows-x86_64-py311.7z
        "Windows": SolverAsset(
            name="FreeCAD",
            url=(
                "https://github.com/FreeCAD/FreeCAD/releases/latest/download/"
                "FreeCAD_1.1.1-Windows-x86_64-py311-installer.exe"
            ),
            archive="FreeCAD_1.1.1-Windows-x86_64-py311-installer.exe",
            binary_glob="**/bin/FreeCAD.exe",
            env_var="PYLCSS_FREECAD_EXE",
            extra_globs={
                "PYLCSS_FREECAD_CMD": "**/bin/FreeCADCmd.exe",
                "PYLCSS_FREECAD_PYTHON": "**/bin/python.exe",
            },
        ),
        "Linux": SolverAsset(
            name="FreeCAD",
            url=(
                "https://github.com/FreeCAD/FreeCAD/releases/latest/download/"
                "FreeCAD_1.1.1-Linux-x86_64-py311.AppImage"
            ),
            archive="FreeCAD_1.1.1-Linux-x86_64-py311.AppImage",
            # AppImages are single-file self-extracting; we don't unzip them,
            # just chmod +x and point at them directly.  install_solver()
            # special-cases the .AppImage suffix below.
            binary_glob="*.AppImage",
            env_var="PYLCSS_FREECAD_EXE",
        ),
        "Darwin": SolverAsset(
            name="FreeCAD",
            url="manual:download https://github.com/FreeCAD/FreeCAD/releases/latest",
            archive="",
            binary_glob="FreeCAD.app/Contents/MacOS/FreeCAD",
            env_var="PYLCSS_FREECAD_EXE",
        ),
    },
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
    if archive.suffix == ".7z":
        # FreeCAD's Windows portable 7z uses LZMA2 + a BCJ2 branch filter,
        # which py7zr does NOT support (it raises
        # UnsupportedCompressionMethodError). The reliable fallback is the
        # 7-Zip CLI (`7z.exe`), which ships with the free 7-Zip install
        # most Windows users already have. Try CLI first; only fall back to
        # py7zr for archives without BCJ2.
        seven_zip = _find_7z_cli()
        if seven_zip:
            print(f"  -> using 7-Zip CLI at {seven_zip}")
            proc = subprocess.run(
                [seven_zip, "x", str(archive), f"-o{dest}", "-y"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            if proc.returncode != 0:
                raise SystemExit(
                    f"7-Zip CLI failed to extract {archive.name}:\n{proc.stdout}"
                )
            return

        # No 7z.exe on the system. Try py7zr; if BCJ2 trips it (likely for
        # FreeCAD), give the user a clear actionable error instead of a
        # cryptic traceback.
        try:
            import py7zr
        except ImportError as exc:
            raise SystemExit(
                f"7z archive {archive.name}: neither 7z.exe nor py7zr available. "
                "Install 7-Zip from https://www.7-zip.org/ (recommended) "
                "or `pip install py7zr` for limited fallback."
            ) from exc
        try:
            with py7zr.SevenZipFile(archive, mode="r") as z:
                z.extractall(dest)
        except Exception as exc:
            raise SystemExit(
                f"py7zr cannot extract {archive.name} ({exc.__class__.__name__}: {exc}).\n"
                "FreeCAD's 7z uses BCJ2 which py7zr doesn't support.\n"
                "FIX: install 7-Zip from https://www.7-zip.org/ (puts 7z.exe on PATH),\n"
                "     then re-run `python scripts/install_solvers.py --only freecad`.\n"
                "ALTERNATIVE: download the FreeCAD installer wizard instead:\n"
                "     https://github.com/FreeCAD/FreeCAD/releases/latest -> "
                "FreeCAD_*-Windows-x86_64-py311-installer.exe\n"
                "     then set PYLCSS_FREECAD_EXE to the installed FreeCAD.exe path."
            ) from exc
        return
    if archive.suffix == ".AppImage":
        # AppImages don't extract: they're a single executable. Just place
        # the file (already downloaded) under dest and mark it executable.
        target = dest / archive.name
        shutil.copy2(archive, target)
        try:
            target.chmod(0o755)
        except Exception:
            pass
        return
    if archive.suffix == ".exe" and "installer" in archive.name.lower():
        # Windows installer wizard (FreeCAD ships one because the portable
        # 7z uses BCJ2 which py7zr can't decode). Drop a copy under dest so
        # `_find_first` can locate it for the user, but don't try to "extract"
        # an .exe -- install_solver() runs the wizard separately.
        target = dest / archive.name
        shutil.copy2(archive, target)
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


def _run_elevated_wait(exe_path: Path, ps_argument_list: List[str]) -> int:
    """Launch ``exe_path`` with admin elevation on Windows + wait for it.

    ``ps_argument_list`` is the list of PowerShell-quoted arguments to pass
    (each element should already be wrapped in single quotes, e.g.
    ``"'/SILENT'"``).  Empty list means "no arguments".

    Returns the installer's exit code, or -1 if PowerShell itself failed
    to launch.

    Why this exists: ``subprocess.run`` cannot trigger a UAC prompt, so
    admin-elevation installers (Inno Setup writing to Program Files)
    raise ``WinError 740`` from CreateProcess.  ``Start-Process -Verb
    RunAs`` is the documented Windows way to request elevation from a
    non-elevated context, ``-Wait`` blocks until the elevated child
    exits, ``-PassThru`` gives us a process handle so we can read its
    ``ExitCode``.
    """
    if platform.system() != "Windows":
        # Non-Windows callers should never hit this path (no installer.exe
        # asset on Linux/macOS), but if they do, fall back to a normal run.
        return subprocess.run([str(exe_path), *(a.strip("'") for a in ps_argument_list)]).returncode

    # PowerShell single-quoted strings: escape any embedded apostrophe by
    # doubling it (PowerShell convention).  Paths with apostrophes are
    # exotic but we shouldn't crash on them.
    safe_path = str(exe_path).replace("'", "''")

    if ps_argument_list:
        argument_clause = " -ArgumentList " + ",".join(ps_argument_list)
    else:
        argument_clause = ""

    # Wrap Start-Process in try/catch so a UAC reject (which raises a
    # System.InvalidOperationException, NOT a non-zero exit code) gets
    # mapped to a real failure code we can act on instead of being
    # swallowed.  1602 = ERROR_INSTALL_USEREXIT, the canonical "user
    # cancelled" value Inno Setup itself uses.
    ps_script = (
        "try {"
        f"  $p = Start-Process -FilePath '{safe_path}'"
        f"{argument_clause} -Verb RunAs -Wait -PassThru;"
        "  if ($null -eq $p) { exit 1602 } else { exit $p.ExitCode }"
        "} catch {"
        "  Write-Error $_;"
        "  exit 1602"
        "}"
    )
    # encoding='utf-8' + errors='replace' so a non-English Windows locale
    # (e.g. cp1252 on German Windows) doesn't crash _readerthread when
    # Inno Setup's stderr contains UTF-8 bytes outside the locale codepage.
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.stderr.strip():
        # Always surface stderr -- previously we only printed it on
        # non-zero returncode, but PowerShell can write warnings even on
        # a successful elevated launch, and we want them visible during
        # debugging.
        print(f"  PowerShell stderr: {proc.stderr.strip()}")
    return proc.returncode


def _find_7z_cli() -> Optional[str]:
    """Locate the system 7-Zip command-line tool.

    Used for FreeCAD's portable 7z release which uses the BCJ2 branch
    filter that py7zr can't decode. Looks at common install locations
    first (Windows), then PATH. Returns None when nothing usable is
    found -- caller falls back to py7zr + a clear error.
    """
    # Explicit env override wins.
    explicit = os.environ.get("PYLCSS_SEVEN_ZIP")
    if explicit and Path(explicit).is_file():
        return explicit

    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ]
        for cand in candidates:
            if Path(cand).is_file():
                return cand

    # PATH lookup (covers Linux/macOS installs + Windows users who put 7z
    # on PATH themselves).
    for name in ("7z", "7zz", "7z.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


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

    # Special case: Windows installer wizard (FreeCAD). After download we
    # run the .exe; it puts the real binaries under Program Files where the
    # main glob below can't reach, so we resolve via _find_installed_program.
    if archive_path.suffix == ".exe" and "installer" in archive_path.name.lower():
        return _run_installer_and_resolve(asset, archive_path)

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


def _run_installer_and_resolve(asset: SolverAsset, installer_path: Path) -> Dict[str, str]:
    """Run a Windows installer wizard then locate the resulting binaries.

    Skips the wizard entirely when FreeCAD is already on the system --
    re-running an Inno Setup installer over a working install is
    annoying (UAC popup every run) and accomplishes nothing.

    For first-time installs we launch the visible wizard via
    PowerShell's ``Start-Process -Verb RunAs`` which triggers UAC; a
    plain ``subprocess.run`` raises ``WinError 740`` because it cannot
    elevate.  After the wizard closes we re-run the same detector to
    pick up the install location (Program Files, AppData, registry,
    or wherever the user pointed it).
    """
    # Pre-flight: if FreeCAD is already detectable on the system, skip
    # the wizard entirely.  This is the path users hit on every re-run
    # of install_solvers.py after the first successful install.
    pre = _detect_installed_freecad(asset, verbose=False)
    if pre:
        print(f"[{asset.name}] already on system at {pre[asset.env_var]} (skipping installer)")
        return pre

    print(f"[{asset.name}] launching installer: {installer_path}")
    print("  A UAC prompt will appear -- accept it.")
    print("  Then click through the installer wizard. PyLCSS waits until it closes,")
    print("  then auto-detects the install location (Program Files, AppData, or registry).")

    # Use the VISIBLE wizard, not silent.  Silent installs have failed
    # invisibly in two ways here:
    #   1. UAC reject -> PowerShell returns 0 anyway
    #   2. Antivirus block -> installer never starts
    # Either way the user sees "no binary found" with no clue what went
    # wrong. A visible wizard gives the user a chance to spot + handle
    # the issue (UAC, AV warning, custom install path).
    rc = _run_elevated_wait(installer_path, [])
    if rc != 0:
        print(
            f"  ! installer wizard exited with code {rc}.\n"
            "    Common causes:\n"
            "      - User declined UAC prompt\n"
            "      - User cancelled the wizard\n"
            "      - Antivirus blocked the installer\n"
            "    If you DID install successfully, re-run this script -- the auto-\n"
            "    detect step below should pick it up."
        )

    resolved = _detect_installed_freecad(asset, verbose=True)
    if resolved:
        return resolved

    raise SystemExit(
        f"[{asset.name}] installer ran but no FreeCAD binary was detected.\n"
        "If you installed FreeCAD to a custom path, set it manually:\n"
        "  setx PYLCSS_FREECAD_EXE \"C:\\path\\to\\FreeCAD.exe\"\n"
        "Or open external_solvers/solver_paths.json and add\n"
        "  \"PYLCSS_FREECAD_EXE\": \"C:/path/to/FreeCAD.exe\"."
    )


def _detect_installed_freecad(asset: SolverAsset, verbose: bool = True) -> Dict[str, str]:
    """Probe the system for an existing FreeCAD install.  Returns an empty
    dict when nothing found; otherwise the same ``{env_var: path}`` map
    a fresh install would produce.

    Lookup order (cheapest first):
      1. Hardcoded Program Files + per-user AppData defaults
      2. Windows uninstall registry under HKLM + HKCU
      3. PATH
      4. Deep recursive scan of common roots (slow, last resort)
    """
    # Probe the standard FreeCAD install layout for binaries. paths.py uses
    # the same list, kept in sync.  Also covers per-user installs in AppData
    # (Inno Setup falls back there if elevation is denied) and reads the
    # Windows uninstall registry as a last resort -- that's the only place
    # custom-directory installs leave a forwarding pointer.
    candidates: List[Path] = [
        Path(r"C:\Program Files\FreeCAD 1.1"),
        Path(r"C:\Program Files\FreeCAD 1.0"),
        Path(r"C:\Program Files\FreeCAD"),
        # Per-user install paths used by Inno Setup when admin is declined.
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\FreeCAD 1.1")),
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\FreeCAD 1.0")),
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\FreeCAD")),
        Path(os.path.expandvars(r"%LOCALAPPDATA%\FreeCAD 1.1")),
        Path(os.path.expandvars(r"%APPDATA%\FreeCAD")),
    ]
    # Registry lookup adds whatever the actual Inno Setup install directory
    # was, even if the user picked something exotic.
    reg_install = _find_freecad_install_from_registry()
    if reg_install:
        candidates.insert(0, reg_install)
        if verbose:
            print(f"  registry hint: {reg_install}")

    resolved: Dict[str, str] = {}
    for root in candidates:
        if not root.is_dir():
            continue
        main = _find_first(root, asset.binary_glob)
        if main is None:
            continue
        resolved[asset.env_var] = str(main)
        for env_var, pattern in asset.extra_globs.items():
            cand = _find_first(root, pattern)
            if cand:
                resolved[env_var] = str(cand)
        if verbose:
            print(f"[{asset.name}] detected install at {root}")
        break

    # PATH next (some installers add bin/ to PATH).
    if not resolved:
        from_path = shutil.which("FreeCAD") or shutil.which("FreeCAD.exe")
        if from_path:
            resolved[asset.env_var] = from_path
            if verbose:
                print(f"[{asset.name}] detected via PATH: {from_path}")

    # Last resort: recursive scan of common install roots. Slow (~10 s),
    # so skip it when the caller asked for a quick probe (pre-install
    # check); rely on the wizard's post-install resolver to invoke this.
    if not resolved and verbose:
        print(f"[{asset.name}] standard paths empty, running deep scan (this can take ~10 s)...")
        roots = [
            Path(r"C:\Program Files"),
            Path(r"C:\Program Files (x86)"),
            Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs")),
            Path(os.path.expandvars(r"%LOCALAPPDATA%")),
            Path(os.path.expandvars(r"%APPDATA%")),
            Path(os.path.expandvars(r"%PROGRAMDATA%")),
            Path(os.path.expandvars(r"%USERPROFILE%")),
        ]
        for root in roots:
            if not root.is_dir():
                continue
            for found in root.glob("**/bin/FreeCAD.exe"):
                # Skip our own staged installer extract dir and any old
                # uninstalled remnants without a real bin/ tree.
                if "external_solvers" in found.parts:
                    continue
                resolved[asset.env_var] = str(found)
                fc_root = found.parent.parent
                for env_var, pattern in asset.extra_globs.items():
                    cand = _find_first(fc_root, pattern)
                    if cand:
                        resolved[env_var] = str(cand)
                print(f"[{asset.name}] detected via deep scan: {found}")
                break
            if resolved:
                break

    return resolved


def _find_freecad_install_from_registry() -> Optional[Path]:
    """Probe the Windows uninstall registry for the FreeCAD install dir.

    Inno Setup writes ``InstallLocation`` under both HKLM and HKCU
    depending on whether the install was system- or per-user-scoped.  We
    scan both, returning the first FreeCAD entry whose ``InstallLocation``
    actually exists on disk (stale registry entries from old uninstalled
    versions are common).
    """
    if platform.system() != "Windows":
        return None
    for hive in ("HKLM", "HKCU"):
        for view in (r"\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                     r"\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall"):
            try:
                proc = subprocess.run(
                    ["reg", "query", f"{hive}{view}", "/s", "/f", "FreeCAD"],
                    capture_output=True, text=True, timeout=10,
                    encoding="utf-8", errors="replace",
                )
            except Exception:
                continue
            if proc.returncode != 0 or not proc.stdout:
                continue
            # Walk the output looking for `InstallLocation    REG_SZ    <path>`.
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("InstallLocation") and "REG_SZ" in line:
                    path = line.split("REG_SZ", 1)[1].strip()
                    p = Path(path)
                    if p.is_dir():
                        return p
    return None


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
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(SOLVERS.keys()),
        help="Install a specific component (can be repeated, e.g. --only ccx --only freecad).",
    )
    parser.add_argument("--all", action="store_true",
                        help="Install every supported component non-interactively.")
    parser.add_argument("--force", action="store_true",
                        help="Reinstall even if already extracted.")
    parser.add_argument("--list", action="store_true",
                        help="Show the URLs that would be downloaded.")
    parser.add_argument(
        "--url-override",
        help="Replace the upstream URL for the selected solver (use with a single --only).",
    )
    args = parser.parse_args(argv)

    plat = _platform_key()

    if args.list:
        for key in sorted(SOLVERS.keys()):
            asset = SOLVERS[key].get(plat)
            if not asset:
                print(f"{key} [{plat}] -> (no asset)")
            elif asset.url.startswith(("apt:", "brew:", "manual:")):
                print(f"{key} [{plat}] -> manual: {asset.url}")
            else:
                print(f"{key} [{plat}] -> {asset.url}")
        return 0

    # Decide what to install:
    #   --only X [--only Y]     -> just those, no prompt
    #   --all                   -> everything, no prompt (CI / unattended use)
    #   (no flag)               -> interactive Y/N per component, so users
    #                              who only want FreeCAD don't get CalculiX
    #                              + OpenRadioss forced on them.
    if args.only:
        keys = list(dict.fromkeys(args.only))  # de-dup, preserve order
    elif args.all:
        keys = list(SOLVERS.keys())
    else:
        keys = _prompt_for_components(plat)
        if not keys:
            print("Nothing selected. Exiting.")
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


def _prompt_for_components(plat: str) -> List[str]:
    """Interactive Y/N per component when the user didn't pass --only / --all.

    Reads one line per component from stdin.  Empty input defaults to N
    (skip) -- safer than defaulting to Y because each install can launch
    a UAC-elevated wizard or a multi-hundred-MB download.
    """
    print("PyLCSS external components (each is optional, PyLCSS opens cleanly without them):")
    print()
    selected: List[str] = []
    for key in SOLVERS.keys():
        asset = SOLVERS[key].get(plat)
        if asset is None:
            print(f"  {key:8s} - no asset for {plat}, skipping")
            continue
        purpose = _component_purpose(key)
        print(f"  {key:8s} - {purpose}")
        try:
            ans = input(f"           install {key}? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        if ans in ("y", "yes"):
            selected.append(key)
        print()
    if selected:
        print("Will install:", ", ".join(selected))
    return selected


def _component_purpose(key: str) -> str:
    """One-line description shown next to each component during the prompt."""
    return {
        "ccx": "CalculiX -- linear static FEA solver (~10 MB download)",
        "radioss": "OpenRadioss -- explicit crash / impact solver (~250 MB)",
        "freecad": "FreeCAD 1.x -- interactive parametric CAD GUI (~1 GB, runs Windows installer wizard with UAC)",
    }.get(key, "external component")


if __name__ == "__main__":
    sys.exit(main())
