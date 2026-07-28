# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12.10",
    [string]$PythonArchiveSha256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3",
    [string]$InnoCompiler = "",
    [string]$OutputBaseFilename = "PyLCSS-2.2.0-Setup-x64"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Candidate
    )

    $parentPath = [System.IO.Path]::GetFullPath($Parent).TrimEnd("\")
    $candidatePath = [System.IO.Path]::GetFullPath($Candidate)
    if (-not $candidatePath.StartsWith(
        "$parentPath\",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing operation outside $parentPath`: $candidatePath"
    }
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$Sha256 = ""
    )

    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination
    }
    if ($Sha256) {
        $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
        if ($actual -ne $Sha256) {
            throw "SHA-256 mismatch for $Destination. Expected $Sha256, got $actual."
        }
    }
}

function Find-InnoCompiler {
    param([string]$ExplicitPath)

    if ($ExplicitPath) {
        if (-not (Test-Path -LiteralPath $ExplicitPath -PathType Leaf)) {
            throw "ISCC.exe was not found at $ExplicitPath"
        }
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }
    $command = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    throw (
        "Inno Setup 6 is required to compile PyLCSS-Setup.exe. " +
        "Install JRSoftware.InnoSetup with winget, then rerun this script."
    )
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\.."))
$buildRoot = Join-Path $repoRoot "build\windows-installer"
$stagingRoot = Join-Path $buildRoot "staging"
$cacheRoot = Join-Path $buildRoot "cache"
$launcherEnvironment = Join-Path $buildRoot "launcher-venv"
$launcherBuild = Join-Path $buildRoot "launcher-build"
$launcherDist = Join-Path $buildRoot "launcher-dist"
$appStage = Join-Path $stagingRoot "app"
$runtimeStage = Join-Path $stagingRoot "runtime"
$launcherStage = Join-Path $stagingRoot "launcher"
$developmentExecutable = Join-Path $repoRoot "PyLCSS.exe"

Assert-ChildPath -Parent $repoRoot -Candidate $buildRoot
Assert-ChildPath -Parent $buildRoot -Candidate $stagingRoot
Assert-ChildPath -Parent $buildRoot -Candidate $launcherBuild
Assert-ChildPath -Parent $buildRoot -Candidate $launcherDist

foreach ($path in @($stagingRoot, $launcherBuild, $launcherDist)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
New-Item -ItemType Directory -Path $appStage -Force | Out-Null
New-Item -ItemType Directory -Path $runtimeStage -Force | Out-Null
New-Item -ItemType Directory -Path $launcherStage -Force | Out-Null
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null

foreach ($fileName in @(
    "LICENSE",
    "NOTICE",
    "README.md",
    "pyproject.toml",
    "requirements.txt"
)) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $fileName) -Destination $appStage
}
foreach ($directoryName in @("pylcss", "data", "data_freecad", "scripts")) {
    $source = Join-Path $repoRoot $directoryName
    if (Test-Path -LiteralPath $source -PathType Container) {
        Copy-Item -LiteralPath $source -Destination $appStage -Recurse
    }
}

# Never ship local bytecode, logs, credentials, assistant state, or generated
# runtime data. These can exist in a developer checkout even when ignored.
$cacheDirectories = Get-ChildItem -LiteralPath $appStage -Recurse -Directory |
    Where-Object { $_.Name -eq "__pycache__" }
foreach ($directory in $cacheDirectories) {
    Assert-ChildPath -Parent $stagingRoot -Candidate $directory.FullName
    Remove-Item -LiteralPath $directory.FullName -Recurse -Force
}
$generatedFiles = Get-ChildItem -LiteralPath $appStage -Recurse -File |
    Where-Object {
        $_.Extension -in @(".pyc", ".pyo", ".log") -or
        $_.Name -in @(".llm_key", "llm_memory.json", "settings.json")
    }
foreach ($file in $generatedFiles) {
    Assert-ChildPath -Parent $stagingRoot -Candidate $file.FullName
    Remove-Item -LiteralPath $file.FullName -Force
}

$pythonArchiveName = "python-$PythonVersion-embed-amd64.zip"
$pythonArchiveCache = Join-Path $cacheRoot $pythonArchiveName
Get-VerifiedDownload `
    -Uri "https://www.python.org/ftp/python/$PythonVersion/$pythonArchiveName" `
    -Destination $pythonArchiveCache `
    -Sha256 $PythonArchiveSha256
Copy-Item -LiteralPath $pythonArchiveCache -Destination $runtimeStage

$pipZipApp = Join-Path $cacheRoot "pip.pyz"
Get-VerifiedDownload `
    -Uri "https://bootstrap.pypa.io/pip/pip.pyz" `
    -Destination $pipZipApp
Copy-Item -LiteralPath $pipZipApp -Destination $runtimeStage

$buildPython = Join-Path $launcherEnvironment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $buildPython -PathType Leaf)) {
    & (Get-Command python).Source -m venv $launcherEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated launcher build environment."
    }
}
& $buildPython -m pip install --disable-pip-version-check --quiet `
    "pyinstaller==6.21.0" "pillow>=11.0"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the launcher build dependencies."
}

$launcherIcon = Join-Path $launcherStage "PyLCSS.ico"
$sourceIcon = Join-Path $repoRoot "pylcss\user_interface\icon.png"
& $buildPython (Join-Path $scriptRoot "create_branding_assets.py") `
    --source $sourceIcon `
    --output-directory $launcherStage
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the Windows application and installer artwork."
}

& $buildPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "PyLCSS" `
    --icon $launcherIcon `
    --version-file (Join-Path $scriptRoot "launcher_version_info.txt") `
    --workpath $launcherBuild `
    --distpath $launcherDist `
    --specpath $buildRoot `
    (Join-Path $scriptRoot "launcher.py")
if ($LASTEXITCODE -ne 0) {
    throw "PyLCSS.exe launcher build failed."
}
Copy-Item `
    -LiteralPath (Join-Path $launcherDist "PyLCSS.exe") `
    -Destination $launcherStage
Copy-Item `
    -LiteralPath (Join-Path $launcherDist "PyLCSS.exe") `
    -Destination $developmentExecutable `
    -Force

$compiler = Find-InnoCompiler -ExplicitPath $InnoCompiler
& $compiler "/DPythonArchiveName=$pythonArchiveName" `
    "/DOutputBaseFilename=$OutputBaseFilename" `
    (Join-Path $scriptRoot "PyLCSS.iss")
if ($LASTEXITCODE -ne 0) {
    throw "The Inno Setup compiler failed."
}

$setupPath = Join-Path $repoRoot "$OutputBaseFilename.exe"
if (-not (Test-Path -LiteralPath $setupPath -PathType Leaf)) {
    throw "The setup compiler completed but $setupPath was not created."
}
$setupHash = (Get-FileHash -LiteralPath $setupPath -Algorithm SHA256).Hash
Write-Host "Created $setupPath"
Write-Host "SHA-256 $setupHash"
