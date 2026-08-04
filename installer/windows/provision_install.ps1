# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot,

    [Parameter(Mandatory = $true)]
    [string]$PythonArchive,

    [Parameter(Mandatory = $true)]
    [string]$PipZipApp,

    [ValidateSet("All", "Runtime", "Bootstrap", "Requirements", "Verify")]
    [string]$Phase = "All"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-PathWithinInstall {
    param([Parameter(Mandatory = $true)][string]$Candidate)

    $root = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd("\")
    $resolved = [System.IO.Path]::GetFullPath($Candidate)
    if (-not $resolved.StartsWith(
        "$root\",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing path outside the PyLCSS installation: $resolved"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE."
    }
}

function Write-InstallStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$State,
        [Parameter(Mandatory = $true)][string]$Message
    )

    $status = [ordered]@{
        stage = $Stage
        state = $State
        message = $Message
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $status | ConvertTo-Json -Depth 2 |
        Set-Content -LiteralPath $statusPath -Encoding utf8
}

$root = [System.IO.Path]::GetFullPath($InstallRoot)
$runtimeRoot = Join-Path $root "runtime"
$pythonRoot = Join-Path $runtimeRoot "python"
$pythonExe = Join-Path $pythonRoot "python.exe"
$applicationRoot = Join-Path $root "app"
$requirements = Join-Path $applicationRoot "requirements.txt"
$requirementsLock = Join-Path $applicationRoot "requirements-windows-py312.lock"
$bootstrapLock = Join-Path $InstallRoot "installer\bootstrap-requirements.lock"
$licenseExporter = Join-Path $applicationRoot "scripts\export_third_party_licenses.py"
$pythonLicenseRoot = Join-Path $applicationRoot "THIRD_PARTY_LICENSES\python-packages"
$logRoot = Join-Path $root "install\logs"
$receiptPath = Join-Path $root "install\installation.json"
$statusPath = Join-Path $root "install\status.json"

Assert-PathWithinInstall -Candidate $runtimeRoot
Assert-PathWithinInstall -Candidate $pythonRoot
Assert-PathWithinInstall -Candidate $pythonLicenseRoot
Assert-PathWithinInstall -Candidate $logRoot
Assert-PathWithinInstall -Candidate $receiptPath
Assert-PathWithinInstall -Candidate $statusPath

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$logPath = Join-Path $logRoot (
    "provision-$($Phase.ToLowerInvariant())-" +
    "$((Get-Date).ToString('yyyyMMdd-HHmmss-fff')).log"
)

Start-Transcript -Path $logPath -Force | Out-Null
try {
    if ($Phase -in @("All", "Runtime")) {
        Write-InstallStatus `
            -Stage "runtime" `
            -State "running" `
            -Message "Preparing the isolated Python runtime."

        if (-not (Test-Path -LiteralPath $PythonArchive -PathType Leaf)) {
            throw "Bundled Python archive is missing: $PythonArchive"
        }
        $archiveName = [System.IO.Path]::GetFileName($PythonArchive)
        if ($archiveName -notmatch '^python-(\d+\.\d+\.\d+)-embed-amd64\.zip$') {
            throw "Could not determine the bundled Python version from $archiveName"
        }
        $desiredPythonVersion = $Matches[1]
        $installedPythonVersion = ""
        if (Test-Path -LiteralPath $pythonExe -PathType Leaf) {
            try {
                $installedPythonVersion = (
                    & $pythonExe --version 2>&1 | Out-String
                ).Trim() -replace '^Python\s+', ''
            }
            catch {
                $installedPythonVersion = ""
            }
        }
        if ($installedPythonVersion -ne $desiredPythonVersion) {
            if (Test-Path -LiteralPath $pythonRoot) {
                Assert-PathWithinInstall -Candidate $pythonRoot
                Remove-Item -LiteralPath $pythonRoot -Recurse -Force
            }
            New-Item -ItemType Directory -Path $pythonRoot -Force | Out-Null
            Expand-Archive `
                -LiteralPath $PythonArchive `
                -DestinationPath $pythonRoot `
                -Force
        }

        $pthFile = Get-ChildItem -LiteralPath $pythonRoot -Filter "python*._pth" |
            Select-Object -First 1
        if ($null -eq $pthFile) {
            throw "The isolated Python path configuration was not found."
        }
        $pthLines = Get-Content -LiteralPath $pthFile.FullName
        $pthLines = $pthLines | ForEach-Object {
            if ($_ -eq "#import site") { "import site" } else { $_ }
        }
        if ($pthLines -notcontains "Lib\site-packages") {
            $pthLines += "Lib\site-packages"
        }
        # CPython's embeddable runtime ignores PYTHONPATH while a ._pth file is
        # active. The application directory must therefore be explicit here.
        if ($pthLines -notcontains $applicationRoot) {
            $pthLines += $applicationRoot
        }
        Set-Content `
            -LiteralPath $pthFile.FullName `
            -Value $pthLines `
            -Encoding ascii

        Write-InstallStatus `
            -Stage "runtime" `
            -State "complete" `
            -Message "The isolated Python runtime is ready."
    }

    if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
        throw "requirements.txt is missing from the installed application."
    }
    if (-not (Test-Path -LiteralPath $requirementsLock -PathType Leaf)) {
        throw "The hashed Windows dependency lock is missing: $requirementsLock"
    }
    if (-not (Test-Path -LiteralPath $bootstrapLock -PathType Leaf)) {
        throw "The hashed bootstrap dependency lock is missing: $bootstrapLock"
    }
    if (-not (Test-Path -LiteralPath $PipZipApp -PathType Leaf)) {
        throw "The bundled pip bootstrap is missing: $PipZipApp"
    }

    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PIP_NO_INPUT = "1"
    $env:PIP_DEFAULT_TIMEOUT = "120"
    $env:PIP_RETRIES = "5"
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONPATH = $applicationRoot

    if ($Phase -in @("All", "Bootstrap")) {
        Write-InstallStatus `
            -Stage "bootstrap" `
            -State "running" `
            -Message "Installing Python packaging tools."
        Invoke-Checked -Executable $pythonExe -Arguments @(
            $PipZipApp,
            "install",
            "--no-warn-script-location",
            "--upgrade",
            "--only-binary=:all:",
            "--require-hashes",
            "--requirement",
            $bootstrapLock
        )
        Write-InstallStatus `
            -Stage "bootstrap" `
            -State "complete" `
            -Message "Python packaging tools are ready."
    }

    if ($Phase -in @("All", "Requirements")) {
        Write-InstallStatus `
            -Stage "requirements" `
            -State "running" `
            -Message "Installing engineering packages. This is a large download."
        Invoke-Checked -Executable $pythonExe -Arguments @(
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "--only-binary=:all:",
            "--require-hashes",
            "--requirement",
            $requirementsLock
        )
        if (-not (Test-Path -LiteralPath $licenseExporter -PathType Leaf)) {
            throw "The third-party license exporter is missing: $licenseExporter"
        }
        Invoke-Checked -Executable $pythonExe -Arguments @(
            $licenseExporter,
            "--output",
            $pythonLicenseRoot
        )
        Write-InstallStatus `
            -Stage "requirements" `
            -State "complete" `
            -Message "Engineering packages and their license notices are installed."
    }

    if ($Phase -in @("All", "Verify")) {
        Write-InstallStatus `
            -Stage "verify" `
            -State "running" `
            -Message "Verifying the PyLCSS runtime."
        Invoke-Checked -Executable $pythonExe -Arguments @(
            "-c",
            "import PySide6, cadquery, pymoto, torch, vtk, pylcss; print('PyLCSS runtime verified')"
        )

        $receipt = [ordered]@{
            product = "PyLCSS"
            version = "2.2.0"
            installed_at = (Get-Date).ToUniversalTime().ToString("o")
            python = (& $pythonExe --version 2>&1 | Out-String).Trim()
            runtime = $pythonRoot
            requirements = $requirements
            requirements_lock = $requirementsLock
        }
        New-Item -ItemType Directory -Path (Split-Path $receiptPath) -Force |
            Out-Null
        $receipt | ConvertTo-Json -Depth 3 |
            Set-Content -LiteralPath $receiptPath -Encoding utf8
        Write-InstallStatus `
            -Stage "verify" `
            -State "complete" `
            -Message "The PyLCSS runtime was verified successfully."
    }
}
catch {
    Write-InstallStatus `
        -Stage ($Phase.ToLowerInvariant()) `
        -State "failed" `
        -Message $_.Exception.Message
    throw
}
finally {
    Stop-Transcript | Out-Null
}
