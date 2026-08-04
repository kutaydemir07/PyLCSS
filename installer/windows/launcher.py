# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Windows launcher and signed GitHub Release updater for PyLCSS.

The launcher is frozen as ``PyLCSS.exe`` at the application root. In a source
checkout it uses the repository's ``.venv`` and live Python sources. In an
installed release it uses the isolated runtime provisioned beside the
application and checks the official GitHub repository for a newer stable
release before starting the GUI.

Updates are always performed by the normal, signed Windows setup program. This
keeps application files, the embedded Python runtime, dependency locks,
shortcuts, and uninstall metadata under one transactional installation path.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid


CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
GITHUB_LATEST_RELEASE_API = (
    "https://api.github.com/repos/kutaydemir07/PyLCSS/releases/latest"
)
GITHUB_RELEASE_PATH = "/kutaydemir07/PyLCSS/releases/download/"
UPDATE_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
UPDATE_DEFER_SECONDS = 24 * 60 * 60
NETWORK_TIMEOUT_SECONDS = 5
MAX_RELEASE_METADATA_BYTES = 1024 * 1024
MAX_CHECKSUM_BYTES = 64 * 1024
MAX_INSTALLER_BYTES = 1024 * 1024 * 1024
FALLBACK_VERSION = "2.2.0"
STARTUP_EVENT_ENV = "PYLCSS_STARTUP_EVENT"
STARTUP_WAIT_MILLISECONDS = 3_000
WAIT_OBJECT_0 = 0x00000000
WAIT_FAILED = 0xFFFFFFFF

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$", re.IGNORECASE)
_PROJECT_VERSION_RE = re.compile(
    r'^version\s*=\s*["\']([^"\']+)["\']\s*$', re.MULTILINE
)
_INSTALLER_RE = re.compile(
    r"^PyLCSS-(\d+\.\d+\.\d+)-Setup-x64\.exe$", re.IGNORECASE
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class UpdateError(RuntimeError):
    """An update could not be safely discovered, downloaded, or started."""


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    installer_name: str
    installer_url: str
    installer_size: int
    checksum_name: str
    checksum_url: str
    checksum_size: int
    release_url: str


def _show_message(title: str, message: str, flags: int) -> int:
    return int(ctypes.windll.user32.MessageBoxW(0, message, title, flags))


def _show_error(message: str) -> None:
    _show_message("PyLCSS could not start", message, 0x10)


def _show_update_error(message: str) -> None:
    _show_message("PyLCSS update", message, 0x10)


def _show_info(message: str) -> None:
    _show_message("PyLCSS update", message, 0x40)


def _ask_to_update(current_version: str, release: ReleaseInfo) -> bool:
    response = _show_message(
        "PyLCSS update available",
        (
            f"PyLCSS {release.version} is available. You currently have "
            f"{current_version}.\n\n"
            "Download and run the official Windows installer now? The setup "
            "program will update PyLCSS and install any new or changed "
            "dependencies.\n\n"
            "Choose No to continue with the current version."
        ),
        0x00000004 | 0x00000040 | 0x00010000,
    )
    return response == 6  # IDYES


def _installation_root() -> Path:
    executable = Path(sys.executable).resolve()
    return executable.parent


def _runtime_layout(root: Path) -> tuple[Path, Path, bool]:
    """Return the app directory, Python executable, and development-mode flag."""
    development_python = root / ".venv" / "Scripts" / "pythonw.exe"
    if development_python.is_file() and (root / "pylcss" / "main.py").is_file():
        return root, development_python, True
    return root / "app", root / "runtime" / "python" / "pythonw.exe", False


def _parse_version(value: str) -> tuple[int, int, int]:
    match = _VERSION_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Unsupported PyLCSS version: {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _is_newer(candidate: str, current: str) -> bool:
    return _parse_version(candidate) > _parse_version(current)


def _read_current_version(root: Path, app_dir: Path) -> str:
    receipt = root / "install" / "installation.json"
    try:
        value = json.loads(receipt.read_text(encoding="utf-8-sig")).get("version")
        if isinstance(value, str):
            _parse_version(value)
            return value.removeprefix("v")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    try:
        project_text = (app_dir / "pyproject.toml").read_text(encoding="utf-8")
        match = _PROJECT_VERSION_RE.search(project_text)
        if match is not None:
            value = match.group(1)
            _parse_version(value)
            return value.removeprefix("v")
    except (OSError, ValueError):
        pass

    return FALLBACK_VERSION


def _validate_release_download_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != "github.com"
        or not parsed.path.casefold().startswith(GITHUB_RELEASE_PATH.casefold())
    ):
        raise UpdateError("The release contains an unexpected download URL.")
    return value


def _read_url(url: str, *, maximum_bytes: int, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "PyLCSS-Windows-Updater",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise UpdateError("The update server returned an unexpectedly large response.")
    return payload


def _fetch_latest_release() -> ReleaseInfo:
    try:
        payload = _read_url(
            GITHUB_LATEST_RELEASE_API,
            maximum_bytes=MAX_RELEASE_METADATA_BYTES,
            accept="application/vnd.github+json",
        )
        document = json.loads(payload.decode("utf-8"))
    except UpdateError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError(
                "No stable PyLCSS GitHub Release is published yet."
            ) from exc
        raise UpdateError(f"GitHub returned HTTP {exc.code}.") from exc
    except Exception as exc:
        raise UpdateError(f"Could not contact GitHub: {exc}") from exc

    if not isinstance(document, dict):
        raise UpdateError("GitHub returned invalid release metadata.")
    if document.get("draft") or document.get("prerelease"):
        raise UpdateError("GitHub did not return a stable published release.")

    tag_name = document.get("tag_name")
    if not isinstance(tag_name, str):
        raise UpdateError("The GitHub release has no version tag.")
    try:
        version_tuple = _parse_version(tag_name)
    except ValueError as exc:
        raise UpdateError(str(exc)) from exc
    version = ".".join(str(part) for part in version_tuple)

    assets = document.get("assets")
    if not isinstance(assets, list):
        raise UpdateError("The GitHub release has no downloadable assets.")

    installer_asset: dict[str, Any] | None = None
    checksum_asset: dict[str, Any] | None = None
    expected_installer_name = f"PyLCSS-{version}-Setup-x64.exe"
    expected_checksum_name = f"{expected_installer_name}.sha256"
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        if asset.get("name") == expected_installer_name:
            installer_asset = asset
        elif asset.get("name") == expected_checksum_name:
            checksum_asset = asset

    if installer_asset is None or checksum_asset is None:
        raise UpdateError(
            "The latest release does not contain the matching Windows setup "
            "program and SHA-256 file."
        )

    name_match = _INSTALLER_RE.fullmatch(expected_installer_name)
    if name_match is None or _parse_version(name_match.group(1)) != version_tuple:
        raise UpdateError("The Windows setup filename does not match the release tag.")

    def asset_value(asset: dict[str, Any], key: str) -> str:
        value = asset.get(key)
        if not isinstance(value, str):
            raise UpdateError(f"A GitHub release asset has no {key} value.")
        return value

    def asset_size(asset: dict[str, Any], maximum: int) -> int:
        value = asset.get("size")
        if not isinstance(value, int) or value <= 0 or value > maximum:
            raise UpdateError("A GitHub release asset has an invalid size.")
        return value

    release_url_value = document.get("html_url")
    release_url = release_url_value if isinstance(release_url_value, str) else ""
    return ReleaseInfo(
        version=version,
        installer_name=expected_installer_name,
        installer_url=_validate_release_download_url(
            asset_value(installer_asset, "browser_download_url")
        ),
        installer_size=asset_size(installer_asset, MAX_INSTALLER_BYTES),
        checksum_name=expected_checksum_name,
        checksum_url=_validate_release_download_url(
            asset_value(checksum_asset, "browser_download_url")
        ),
        checksum_size=asset_size(checksum_asset, MAX_CHECKSUM_BYTES),
        release_url=release_url,
    )


def _download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    maximum_bytes: int,
) -> None:
    if expected_size <= 0 or expected_size > maximum_bytes:
        raise UpdateError("The update asset has an invalid advertised size.")

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "PyLCSS-Windows-Updater",
        },
    )
    partial = destination.with_name(f"{destination.name}.part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with urllib.request.urlopen(
            request, timeout=NETWORK_TIMEOUT_SECONDS
        ) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise UpdateError("The downloaded update is unexpectedly large.")
                output.write(chunk)
        if total != expected_size:
            raise UpdateError(
                f"The update download was incomplete (expected {expected_size} "
                f"bytes, received {total})."
            )
        os.replace(partial, destination)
    except UpdateError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"Could not download the update: {exc}") from exc


def _parse_checksum(payload: str, expected_filename: str) -> str:
    match = re.fullmatch(
        r"\s*([0-9a-fA-F]{64})\s+[ *]?([^\r\n]+)\s*", payload
    )
    if match is None or Path(match.group(2).strip()).name != expected_filename:
        raise UpdateError("The release SHA-256 file is invalid.")
    return match.group(1).casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticode_details(path: Path) -> dict[str, str]:
    """Return normalized Authenticode status details for a Windows file."""
    powershell = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    if not powershell.is_file():
        raise UpdateError("Windows PowerShell is required to verify the update signer.")

    script = (
        "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); "
        "$s = Get-AuthenticodeSignature -LiteralPath "
        "$env:PYLCSS_SIGNATURE_FILE; "
        "$thumb = if ($null -ne $s.SignerCertificate) "
        "{ $s.SignerCertificate.Thumbprint } else { '' }; "
        "$subject = if ($null -ne $s.SignerCertificate) "
        "{ $s.SignerCertificate.Subject } else { '' }; "
        "[pscustomobject]@{Status=[string]$s.Status;Thumbprint=$thumb;"
        "Subject=$subject} | "
        "ConvertTo-Json -Compress"
    )
    try:
        signature_environment = os.environ.copy()
        signature_environment["PYLCSS_SIGNATURE_FILE"] = str(path)
        result = subprocess.run(
            [
                str(powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=CREATE_NO_WINDOW,
            env=signature_environment,
        )
        details = json.loads(result.stdout.strip())
    except Exception as exc:
        raise UpdateError(f"Could not verify the update's digital signature: {exc}") from exc
    if not isinstance(details, dict):
        raise UpdateError("Windows returned invalid digital-signature details.")
    return {
        "status": str(details.get("Status", "")),
        "thumbprint": str(details.get("Thumbprint", "")),
        "subject": str(details.get("Subject", "")),
        "returncode": str(result.returncode),
    }


def _verify_authenticode(path: Path) -> str:
    """Require a trusted signature and, when possible, the current publisher."""
    details = _authenticode_details(path)
    if (
        details["returncode"] != "0"
        or details["status"] != "Valid"
        or not details["thumbprint"]
        or not details["subject"]
    ):
        raise UpdateError(
            "The downloaded setup program does not have a valid, trusted "
            "Authenticode signature. It will not be run."
        )

    # A signed installed launcher establishes the expected publisher identity.
    # Matching the certificate subject allows normal certificate renewal while
    # preventing an installer signed by an unrelated trusted publisher.
    if getattr(sys, "frozen", False):
        try:
            launcher_details = _authenticode_details(Path(sys.executable).resolve())
        except UpdateError:
            launcher_details = {}
        expected_subject = (
            launcher_details.get("subject")
            if launcher_details.get("status") == "Valid"
            else ""
        )
        if expected_subject and details["subject"] != expected_subject:
            raise UpdateError(
                "The setup program is signed by a different publisher than the "
                "installed PyLCSS launcher. It will not be run."
            )
    return details["thumbprint"]


def _update_cache_root(version: str) -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path(tempfile.gettempdir())
    return base / "PyLCSS" / "updates" / version


def _prepare_installer(release: ReleaseInfo) -> tuple[Path, str]:
    cache = _update_cache_root(release.version)
    installer = cache / release.installer_name
    checksum_file = cache / release.checksum_name
    _download_file(
        release.checksum_url,
        checksum_file,
        expected_size=release.checksum_size,
        maximum_bytes=MAX_CHECKSUM_BYTES,
    )
    try:
        expected_hash = _parse_checksum(
            checksum_file.read_text(encoding="ascii"), release.installer_name
        )
    except (OSError, UnicodeError) as exc:
        raise UpdateError(f"Could not read the release SHA-256 file: {exc}") from exc

    if not installer.is_file() or installer.stat().st_size != release.installer_size:
        installer.unlink(missing_ok=True)
        _download_file(
            release.installer_url,
            installer,
            expected_size=release.installer_size,
            maximum_bytes=MAX_INSTALLER_BYTES,
        )

    actual_hash = _sha256(installer)
    if actual_hash != expected_hash:
        installer.unlink(missing_ok=True)
        raise UpdateError(
            "The downloaded setup program failed SHA-256 verification. It was deleted."
        )
    signer_thumbprint = _verify_authenticode(installer)
    return installer, signer_thumbprint


def _state_path(root: Path) -> Path:
    return root / "install" / "update-state.json"


def _user_state_path(root: Path) -> Path:
    root_key = hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:16]
    return _update_cache_root("state") / f"installation-{root_key}.json"


def _load_update_state(root: Path) -> dict[str, Any]:
    for path in (_state_path(root), _user_state_path(root)):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(value, dict):
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return {}


def _save_update_state(root: Path, state: dict[str, Any]) -> None:
    for path in (_state_path(root), _user_state_path(root)):
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, path)
            return
        except OSError:
            temporary.unlink(missing_ok=True)


def _write_update_log(root: Path, message: str) -> None:
    paths = (
        root / "install" / "logs" / "updater.log",
        _user_state_path(root).parent / "updater.log",
    )
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and path.stat().st_size > 512 * 1024:
                previous = path.with_suffix(".previous.log")
                previous.unlink(missing_ok=True)
                path.replace(previous)
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f"{timestamp} {message}\n")
            return
        except OSError:
            continue


def _automatic_updates_disabled() -> bool:
    value = os.environ.get("PYLCSS_DISABLE_UPDATE_CHECK", "")
    return value.strip().casefold() in _TRUE_VALUES


def _update_check_due(
    state: dict[str, Any], current_version: str, now: float, force: bool
) -> bool:
    if force:
        return True
    deferred_until = state.get("deferred_until")
    deferred_version = state.get("deferred_version")
    if (
        isinstance(deferred_until, (int, float))
        and deferred_until > now
        and isinstance(deferred_version, str)
    ):
        try:
            if _is_newer(deferred_version, current_version):
                return False
        except ValueError:
            pass

    last_checked_at = state.get("last_checked_at")
    latest_version = state.get("latest_version")
    if not isinstance(last_checked_at, (int, float)):
        return True
    if now - last_checked_at >= UPDATE_CHECK_INTERVAL_SECONDS:
        return True
    if isinstance(latest_version, str):
        try:
            return _is_newer(latest_version, current_version)
        except ValueError:
            return True
    return False


def _maybe_start_update(
    root: Path, app_dir: Path, *, force: bool = False
) -> bool:
    """Check, prompt, and start setup. Return True when setup was launched."""
    current_version = _read_current_version(root, app_dir)
    state = _load_update_state(root)
    now = time.time()
    if not _update_check_due(state, current_version, now, force):
        return False

    try:
        release = _fetch_latest_release()
    except UpdateError as exc:
        state["last_checked_at"] = now
        state["last_error"] = str(exc)
        state.pop("latest_version", None)
        state.pop("release_url", None)
        _save_update_state(root, state)
        _write_update_log(root, f"Update check failed: {exc}")
        if force:
            _show_update_error(str(exc))
        return False

    state.update(
        {
            "last_checked_at": now,
            "latest_version": release.version,
            "release_url": release.release_url,
        }
    )
    state.pop("last_error", None)
    _save_update_state(root, state)
    try:
        update_available = _is_newer(release.version, current_version)
    except ValueError as exc:
        _write_update_log(root, f"Update version comparison failed: {exc}")
        return False

    if not update_available:
        if force:
            _show_info(f"PyLCSS {current_version} is up to date.")
        return False

    if not _ask_to_update(current_version, release):
        state["deferred_version"] = release.version
        state["deferred_until"] = now + UPDATE_DEFER_SECONDS
        _save_update_state(root, state)
        return False

    try:
        installer, signer_thumbprint = _prepare_installer(release)
        subprocess.Popen(
            [str(installer), "/CLOSEAPPLICATIONS", f"/DIR={root}"],
            cwd=str(installer.parent),
            creationflags=CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except (OSError, UpdateError) as exc:
        _write_update_log(root, f"Update preparation failed: {exc}")
        _show_update_error(
            f"PyLCSS {release.version} could not be installed.\n\n{exc}\n\n"
            "The current version will start normally."
        )
        return False

    state.pop("deferred_version", None)
    state.pop("deferred_until", None)
    state["installer_started_at"] = time.time()
    state["installer_version"] = release.version
    _save_update_state(root, state)
    _write_update_log(
        root,
        f"Started installer for {release.version}; signer SHA-1 {signer_thumbprint}.",
    )
    return True


def _split_launcher_arguments(arguments: list[str]) -> tuple[list[str], bool, bool]:
    application_arguments: list[str] = []
    skip_update = False
    force_update = False
    for argument in arguments:
        if argument == "--skip-update-check":
            skip_update = True
        elif argument == "--check-for-updates":
            force_update = True
        else:
            application_arguments.append(argument)
    return application_arguments, skip_update, force_update


def _close_splash() -> None:
    """Close PyInstaller's optional splash without requiring it in source mode."""
    try:
        import pyi_splash  # type: ignore[import-not-found]

        if pyi_splash.is_alive():
            pyi_splash.close()
    except (ImportError, RuntimeError):
        pass


def _create_startup_event() -> tuple[str, int] | None:
    """Create the event that the Qt process sets after showing its main window."""
    name = f"Local\\PyLCSS-Startup-{uuid.uuid4()}"
    create_event = ctypes.windll.kernel32.CreateEventW
    create_event.argtypes = [
        ctypes.c_void_p,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.LPCWSTR,
    ]
    create_event.restype = ctypes.wintypes.HANDLE
    handle = create_event(None, True, False, name)
    if not handle:
        return None
    return name, int(handle)


def _close_startup_event(handle: int) -> None:
    close_handle = ctypes.windll.kernel32.CloseHandle
    close_handle.argtypes = [ctypes.wintypes.HANDLE]
    close_handle.restype = ctypes.wintypes.BOOL
    close_handle(handle)


def _wait_for_application_ready(process: subprocess.Popen[Any], handle: int) -> None:
    """Keep the splash visible until Qt signals readiness, exits, or times out."""
    wait_for_single_object = ctypes.windll.kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD]
    wait_for_single_object.restype = ctypes.wintypes.DWORD
    started = time.monotonic()
    try:
        while (time.monotonic() - started) * 1000 < STARTUP_WAIT_MILLISECONDS:
            result = wait_for_single_object(handle, 100)
            if result in (WAIT_OBJECT_0, WAIT_FAILED) or process.poll() is not None:
                break
    finally:
        _close_startup_event(handle)
        _close_splash()


def main() -> int:
    root = _installation_root()
    app_dir, pythonw, development_mode = _runtime_layout(root)
    application_arguments, skip_update, force_update = _split_launcher_arguments(
        sys.argv[1:]
    )

    if not pythonw.is_file():
        _show_error(
            "No PyLCSS Python runtime was found.\n\n"
            "For development, create the repository .venv and install "
            "requirements.\nFor an installed copy, repair or reinstall PyLCSS."
        )
        return 2
    if not (app_dir / "pylcss" / "main.py").is_file():
        _show_error(
            "The PyLCSS application files are missing.\n\n"
            "Repair or reinstall PyLCSS, then try again."
        )
        return 3

    if (
        not development_mode
        and not skip_update
        and not _automatic_updates_disabled()
        and _maybe_start_update(root, app_dir, force=force_update)
    ):
        _close_splash()
        return 0

    environment = os.environ.copy()
    current_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(app_dir) if not current_path else f"{app_dir}{os.pathsep}{current_path}"
    )
    environment["PYTHONNOUSERSITE"] = "1"
    if development_mode:
        environment["PYLCSS_PROJECT_ROOT"] = str(root)
    else:
        environment["PYLCSS_INSTALL_ROOT"] = str(root)

    startup_event = _create_startup_event()
    if startup_event is not None:
        environment[STARTUP_EVENT_ENV] = startup_event[0]

    try:
        process = subprocess.Popen(
            [str(pythonw), "-m", "pylcss.main", *application_arguments],
            cwd=str(app_dir),
            env=environment,
            creationflags=CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except OSError as exc:
        if startup_event is not None:
            _close_startup_event(startup_event[1])
        _close_splash()
        _show_error(f"Windows could not launch PyLCSS.\n\n{exc}")
        return 4
    if startup_event is not None:
        _wait_for_application_ready(process, startup_event[1])
    else:
        _close_splash()

    exit_code = process.poll()
    if exit_code is not None and exit_code != 0:
        log_file = app_dir / "pylcss.log"
        log_hint = f"\n\nSee log file:\n{log_file}" if log_file.is_file() else ""
        _show_error(f"PyLCSS exited unexpectedly (exit code {exit_code}).{log_hint}")
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
