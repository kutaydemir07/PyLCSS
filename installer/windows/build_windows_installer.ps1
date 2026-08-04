# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12.10",
    [string]$PythonArchiveSha256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3",
    [string]$PipZipAppSha256 = "6ddc3444b803a48d83ccf1c4ad846717b42c8ffc9d74713a53ae829a97201365",
    [string]$InnoCompiler = "",
    [string]$OutputBaseFilename = "PyLCSS-2.2.0-Setup-x64",
    [string]$SigningCertificateThumbprint = $env:PYLCSS_SIGNING_CERT_SHA1,
    [string]$TimestampServer = "http://timestamp.digicert.com",
    [switch]$RequireSignature
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
        [Parameter(Mandatory = $true)][string]$Sha256
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $cachedHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
        if ($cachedHash -ne $Sha256) {
            Remove-Item -LiteralPath $Destination -Force
        }
    }
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination
    }
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) {
        throw "SHA-256 mismatch for $Destination. Expected $Sha256, got $actual."
    }
}

function Copy-ZipNotices {
    param(
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$ExactEntry = ""
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        $entries = $zip.Entries | Where-Object {
            -not $_.FullName.EndsWith("/") -and (
                ($ExactEntry -and $_.FullName -eq $ExactEntry) -or
                (-not $ExactEntry -and (
                    $_.Name -match "^(?i:LICENSE|LICENCE|COPYING|NOTICE|AUTHORS|COPYRIGHT)" -or
                    $_.FullName -eq "pip/_vendor/vendor.txt"
                ))
            )
        }
        if (-not $entries) {
            throw "No matching license material was found in $Archive"
        }
        foreach ($entry in $entries) {
            $relativeName = if ($ExactEntry) {
                $entry.Name
            } else {
                $entry.FullName.Replace("/", "\")
            }
            $target = Join-Path $Destination $relativeName
            Assert-ChildPath -Parent $Destination -Candidate $target
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) `
                -Force | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
                $entry,
                $target,
                $true
            )
        }
    }
    finally {
        $zip.Dispose()
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

function Find-SignTool {
    $command = Get-Command "signtool.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $kitsRoot = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
    if (Test-Path -LiteralPath $kitsRoot -PathType Container) {
        $candidate = Get-ChildItem -LiteralPath $kitsRoot -Recurse -File `
            -Filter "signtool.exe" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }
    throw "signtool.exe was not found. Install the Windows SDK signing tools."
}

function Invoke-CodeSign {
    param(
        [Parameter(Mandatory = $true)][string]$SignTool,
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $true)][string]$CertificateThumbprint,
        [Parameter(Mandatory = $true)][string]$TimestampUri
    )

    & $SignTool sign /sha1 $CertificateThumbprint /fd SHA256 `
        /tr $TimestampUri /td SHA256 $File
    if ($LASTEXITCODE -ne 0) {
        throw "Authenticode signing failed for $File"
    }
    & $SignTool verify /pa /all $File
    if ($LASTEXITCODE -ne 0) {
        throw "Authenticode verification failed for $File"
    }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\.."))
$buildRoot = Join-Path $repoRoot "build\windows-installer"
$stagingRoot = Join-Path $buildRoot "staging"
$cacheRoot = Join-Path $buildRoot "cache"
$launcherEnvironment = Join-Path $buildRoot "launcher-venv"
$launcherBuild = Join-Path $buildRoot "launcher-build"
$launcherDist = Join-Path $buildRoot "launcher-dist"
$installerOutputDir = Join-Path $buildRoot "output"
$appStage = Join-Path $stagingRoot "app"
$runtimeStage = Join-Path $stagingRoot "runtime"
$launcherStage = Join-Path $stagingRoot "launcher"
$bundledLicenseStage = Join-Path $appStage "THIRD_PARTY_LICENSES\bundled"
$commonLicenseStage = Join-Path $appStage "THIRD_PARTY_LICENSES\common"
$licenseCacheRoot = Join-Path $cacheRoot "licenses"
$developmentExecutable = Join-Path $repoRoot "PyLCSS.exe"
$signTool = $null
if ($SigningCertificateThumbprint) {
    $signTool = Find-SignTool
} elseif ($RequireSignature) {
    throw (
        "A release signature is required. Pass -SigningCertificateThumbprint " +
        "or set PYLCSS_SIGNING_CERT_SHA1."
    )
} else {
    Write-Warning (
        "Building unsigned artifacts. Public releases must use -RequireSignature " +
        "with a trusted Authenticode certificate."
    )
}

Assert-ChildPath -Parent $repoRoot -Candidate $buildRoot
Assert-ChildPath -Parent $buildRoot -Candidate $stagingRoot
Assert-ChildPath -Parent $buildRoot -Candidate $launcherBuild
Assert-ChildPath -Parent $buildRoot -Candidate $launcherDist
Assert-ChildPath -Parent $buildRoot -Candidate $installerOutputDir

foreach ($path in @($stagingRoot, $launcherBuild, $launcherDist, $installerOutputDir)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
New-Item -ItemType Directory -Path $appStage -Force | Out-Null
New-Item -ItemType Directory -Path $runtimeStage -Force | Out-Null
New-Item -ItemType Directory -Path $launcherStage -Force | Out-Null
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
New-Item -ItemType Directory -Path $bundledLicenseStage -Force | Out-Null
New-Item -ItemType Directory -Path $commonLicenseStage -Force | Out-Null
New-Item -ItemType Directory -Path $licenseCacheRoot -Force | Out-Null

foreach ($fileName in @(
    "LICENSE",
    "NOTICE",
    "PRIVACY.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-windows-py312.lock"
)) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $fileName) -Destination $appStage
}
Copy-Item -LiteralPath (Join-Path $repoRoot "LICENSE") `
    -Destination (Join-Path $appStage "LICENSE.txt")
Copy-Item -LiteralPath (Join-Path $repoRoot "NOTICE") `
    -Destination (Join-Path $appStage "THIRD_PARTY_NOTICES.txt")
foreach ($directoryName in @("pylcss", "data")) {
    $source = Join-Path $repoRoot $directoryName
    if (Test-Path -LiteralPath $source -PathType Container) {
        Copy-Item -LiteralPath $source -Destination $appStage -Recurse
    }
}
$scriptStage = Join-Path $appStage "scripts"
New-Item -ItemType Directory -Path $scriptStage -Force | Out-Null
foreach ($scriptName in @(
    "install_solvers.py",
    "export_third_party_licenses.py"
)) {
    Copy-Item `
        -LiteralPath (Join-Path $repoRoot "scripts\$scriptName") `
        -Destination $scriptStage
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
        $_.Extension -in @(".pyc", ".pyo", ".log", ".h5", ".hdf5", ".tmp", ".temp") -or
        $_.Name -like "*.cad.0x*" -or
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
Copy-ZipNotices `
    -Archive $pythonArchiveCache `
    -Destination (Join-Path $bundledLicenseStage "CPython-$PythonVersion") `
    -ExactEntry "LICENSE.txt"

$pipZipApp = Join-Path $cacheRoot "pip.pyz"
Get-VerifiedDownload `
    -Uri "https://bootstrap.pypa.io/pip/pip.pyz" `
    -Destination $pipZipApp `
    -Sha256 $PipZipAppSha256
Copy-Item -LiteralPath $pipZipApp -Destination $runtimeStage
Copy-ZipNotices `
    -Archive $pipZipApp `
    -Destination (Join-Path $bundledLicenseStage "pip-zipapp")

$spdxTextRoot = "https://raw.githubusercontent.com/spdx/license-list-data/v3.27.0/text"
$commonLicenses = @(
    @("GPL-2.0.txt", "$spdxTextRoot/GPL-2.0-only.txt", "aaf135472f81c5b4a0dca9367e5bb5e9750032b5bebe5442b36e4c0a47430df3"),
    @("GPL-3.0.txt", "$spdxTextRoot/GPL-3.0-only.txt", "fb981668c18a279e285fc4d83fba1e836cc84dd4daa73c9697d3cfd2d8aca6e0"),
    @("AGPL-3.0.txt", "$spdxTextRoot/AGPL-3.0-or-later.txt", "d8a6cc31abc16b6748c7a21f21611f5a1ec33f67d22ca23d7da1c19b95496bee"),
    @("LGPL-2.1.txt", "$spdxTextRoot/LGPL-2.1-only.txt", "5749785c8bdefafcb5d798270ed0a967036fe2ca63dcedade1627565dfef81d2"),
    @("LGPL-3.0.txt", "$spdxTextRoot/LGPL-3.0-only.txt", "996af0513df21f7496288951c41428a03c174e9e4a9d63665c57d670f845ccb1"),
    @("Apache-2.0.txt", "$spdxTextRoot/Apache-2.0.txt", "074e6e32c86a4c0ef8b3ed25b721ca23aca83df277cd88106ef7177c354615ff"),
    @("OFL-1.1.txt", "$spdxTextRoot/OFL-1.1.txt", "8eea8287e5876b539670cadb82e99f9a7afddec6f6730811be1daf25d2e9bcfd"),
    @("CC-BY-4.0.txt", "$spdxTextRoot/CC-BY-4.0.txt", "d557539df68e771cc1eedcc91d13f70fca930e508d11eedcafa4b15db49e3744")
)
foreach ($license in $commonLicenses) {
    $cachedLicense = Join-Path $licenseCacheRoot $license[0]
    Get-VerifiedDownload `
        -Uri $license[1] `
        -Destination $cachedLicense `
        -Sha256 $license[2]
    Copy-Item -LiteralPath $cachedLicense -Destination $commonLicenseStage
}

$buildPython = Join-Path $launcherEnvironment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $buildPython -PathType Leaf)) {
    & (Get-Command python).Source -m venv $launcherEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated launcher build environment."
    }
}
& $buildPython -m pip install --disable-pip-version-check --quiet `
    --only-binary=:all: "pyinstaller==6.21.0" "pillow==12.3.0"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the launcher build dependencies."
}

$launcherIcon = Join-Path $launcherStage "PyLCSS.ico"
$launcherSplash = Join-Path $launcherStage "PyLCSS-splash.png"
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
    --splash $launcherSplash `
    --version-file (Join-Path $scriptRoot "launcher_version_info.txt") `
    --workpath $launcherBuild `
    --distpath $launcherDist `
    --specpath $buildRoot `
    (Join-Path $scriptRoot "launcher.py")
if ($LASTEXITCODE -ne 0) {
    throw "PyLCSS.exe launcher build failed."
}
$builtLauncher = Join-Path $launcherDist "PyLCSS.exe"
if ($signTool) {
    Invoke-CodeSign `
        -SignTool $signTool `
        -File $builtLauncher `
        -CertificateThumbprint $SigningCertificateThumbprint `
        -TimestampUri $TimestampServer
}
$launcherSitePackages = Join-Path $launcherEnvironment "Lib\site-packages"
$pyInstallerLicense = Get-ChildItem `
    -LiteralPath $launcherSitePackages `
    -Directory `
    -Filter "pyinstaller-*.dist-info" |
    Select-Object -First 1
$pyInstallerCopying = Join-Path $pyInstallerLicense.FullName "licenses\COPYING.txt"
if (-not (Test-Path -LiteralPath $pyInstallerCopying -PathType Leaf)) {
    throw "The PyInstaller license file was not found at $pyInstallerCopying"
}
Copy-Item -LiteralPath $pyInstallerCopying `
    -Destination (Join-Path $bundledLicenseStage "PyInstaller-6.21.0-COPYING.txt")
$pipDistribution = Get-ChildItem `
    -LiteralPath $launcherSitePackages `
    -Directory `
    -Filter "pip-*.dist-info" |
    Select-Object -First 1
$pipLicense = Get-ChildItem `
    -LiteralPath $pipDistribution.FullName `
    -Recurse `
    -File `
    -Filter "LICENSE.txt" |
    Select-Object -First 1
if (-not $pipLicense) {
    throw "The pip license file was not found at $pipLicense"
}
Copy-Item -LiteralPath $pipLicense.FullName `
    -Destination (Join-Path $bundledLicenseStage "pip-zipapp\pip-LICENSE.txt")
Copy-Item `
    -LiteralPath $builtLauncher `
    -Destination $launcherStage
Copy-Item `
    -LiteralPath $builtLauncher `
    -Destination $developmentExecutable `
    -Force

New-Item -ItemType Directory -Path $installerOutputDir -Force | Out-Null
$compiler = Find-InnoCompiler -ExplicitPath $InnoCompiler
$innoLicense = Join-Path (Split-Path -Parent $compiler) "license.txt"
if (Test-Path -LiteralPath $innoLicense -PathType Leaf) {
    Copy-Item -LiteralPath $innoLicense `
        -Destination (Join-Path $bundledLicenseStage "Inno-Setup-LICENSE.txt")
} else {
    Write-Warning "Inno Setup license.txt was not found beside $compiler"
}
$maxAttempts = 3
$isccSuccess = $false

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    & $compiler "/DPythonArchiveName=$pythonArchiveName" `
        "/DOutputBaseFilename=$OutputBaseFilename" `
        "/O$installerOutputDir" `
        (Join-Path $scriptRoot "PyLCSS.iss")
    if ($LASTEXITCODE -eq 0) {
        $isccSuccess = $true
        break
    }
    if ($attempt -lt $maxAttempts) {
        Write-Warning "Inno Setup compiler attempt $attempt failed. Retrying in 3 seconds..."
        Start-Sleep -Seconds 3
    }
}

if (-not $isccSuccess) {
    throw "The Inno Setup compiler failed after $maxAttempts attempts."
}

$compiledSetup = Join-Path $installerOutputDir "$OutputBaseFilename.exe"
if ($signTool) {
    Invoke-CodeSign `
        -SignTool $signTool `
        -File $compiledSetup `
        -CertificateThumbprint $SigningCertificateThumbprint `
        -TimestampUri $TimestampServer
}
$setupPath = Join-Path $repoRoot "$OutputBaseFilename.exe"
Copy-Item -LiteralPath $compiledSetup -Destination $setupPath -Force

if (-not (Test-Path -LiteralPath $setupPath -PathType Leaf)) {
    throw "The setup compiler completed but $setupPath was not created."
}
$setupHash = (Get-FileHash -LiteralPath $setupPath -Algorithm SHA256).Hash
$checksumPath = "$setupPath.sha256"
Set-Content `
    -LiteralPath $checksumPath `
    -Value "$($setupHash.ToLowerInvariant())  $([System.IO.Path]::GetFileName($setupPath))" `
    -Encoding ascii
Write-Host "Created $setupPath"
Write-Host "SHA-256 $setupHash"
Write-Host "Checksum $checksumPath"
