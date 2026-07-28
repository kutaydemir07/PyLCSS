; Copyright (c) 2026 Kutay Demir.
; Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

#define AppName "PyLCSS"
#define AppVersion "2.2.0"
#define AppPublisher "Kutay Demir"
#define StagingRoot "..\..\build\windows-installer\staging"
#ifndef PythonArchiveName
#define PythonArchiveName "python-3.12.10-embed-amd64.zip"
#endif
#ifndef OutputBaseFilename
#define OutputBaseFilename "PyLCSS-2.2.0-Setup-x64"
#endif

[Setup]
AppId={{E4215032-255A-42DE-82F3-7E4ABF3C4D22}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\PyLCSS
DisableDirPage=no
DefaultGroupName=PyLCSS
DisableProgramGroupPage=yes
OutputDir=..\..
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
SetupIconFile={#StagingRoot}\launcher\PyLCSS.ico
WizardImageFile={#StagingRoot}\launcher\PyLCSS-wizard.png
WizardSmallImageFile={#StagingRoot}\launcher\PyLCSS-wizard-small.png
SetupLogging=yes
UninstallDisplayIcon={app}\PyLCSS.exe
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany={#AppPublisher}
VersionInfoDescription=PyLCSS Engineering Design Platform Setup
VersionInfoCopyright=Copyright (c) 2026 Kutay Demir
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
LicenseFile={#StagingRoot}\app\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "solvers"; Description: "Engineering solver backends"; GroupDescription: "Optional components:"; Flags: checkablealone
Name: "solvers\calculix"; Description: "CalculiX structural FEA solver"; Flags: checkedonce
Name: "solvers\radioss"; Description: "OpenRadioss crash and impact solver"; Flags: checkedonce
Name: "freecad"; Description: "FreeCAD interactive CAD authoring backend (large download, separate UAC prompt)"; GroupDescription: "Optional components:"

[Files]
Source: "{#StagingRoot}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StagingRoot}\launcher\PyLCSS.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingRoot}\runtime\{#PythonArchiveName}"; DestDir: "{tmp}"; Flags: ignoreversion deleteafterinstall
Source: "{#StagingRoot}\runtime\pip.pyz"; DestDir: "{tmp}"; Flags: ignoreversion deleteafterinstall
Source: "provision_install.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{group}\PyLCSS"; Filename: "{app}\PyLCSS.exe"; WorkingDir: "{app}\app"
Name: "{autodesktop}\PyLCSS"; Filename: "{app}\PyLCSS.exe"; WorkingDir: "{app}\app"; Tasks: desktopicon

[Run]
Filename: "{app}\PyLCSS.exe"; Description: "Launch PyLCSS"; WorkingDir: "{app}\app"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\app\external_solvers"
Type: filesandordirs; Name: "{app}\install"

[Code]
function RunAndWait(
  const Executable: String;
  const Parameters: String;
  const Description: String;
  const Required: Boolean
): Boolean;
var
  ResultCode: Integer;
begin
  WizardForm.StatusLabel.Caption := Description;
  WizardForm.StatusLabel.Update;
  Result :=
    Exec(Executable, Parameters, ExpandConstant('{app}\app'), SW_HIDE,
      ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result then
  begin
    if Required then
      RaiseException(
        Description + ' failed. See the setup log and ' +
        ExpandConstant('{app}\install\logs') + ' for details.')
    else
      MsgBox(
        Description + ' did not complete. PyLCSS itself is installed; ' +
        'the component can be installed later from the application folder.',
        mbError, MB_OK);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  PowerShell: String;
  PythonExe: String;
  InstallerScript: String;
begin
  if CurStep <> ssPostInstall then
    exit;

  PowerShell := ExpandConstant(
    '{sys}\WindowsPowerShell\v1.0\powershell.exe');
  RunAndWait(
    PowerShell,
    '-NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\installer\provision_install.ps1') +
      '" -InstallRoot "' + ExpandConstant('{app}') +
      '" -PythonArchive "' +
      ExpandConstant('{tmp}\{#PythonArchiveName}') +
      '" -PipZipApp "' + ExpandConstant('{tmp}\pip.pyz') +
      '" -Phase Runtime',
    'Preparing the isolated Python runtime...',
    True);

  RunAndWait(
    PowerShell,
    '-NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\installer\provision_install.ps1') +
      '" -InstallRoot "' + ExpandConstant('{app}') +
      '" -PythonArchive "' +
      ExpandConstant('{tmp}\{#PythonArchiveName}') +
      '" -PipZipApp "' + ExpandConstant('{tmp}\pip.pyz') +
      '" -Phase Bootstrap',
    'Installing Python packaging tools...',
    True);

  RunAndWait(
    PowerShell,
    '-NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\installer\provision_install.ps1') +
      '" -InstallRoot "' + ExpandConstant('{app}') +
      '" -PythonArchive "' +
      ExpandConstant('{tmp}\{#PythonArchiveName}') +
      '" -PipZipApp "' + ExpandConstant('{tmp}\pip.pyz') +
      '" -Phase Requirements',
    'Installing engineering packages (large download; typically 10-20 minutes)...',
    True);

  RunAndWait(
    PowerShell,
    '-NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\installer\provision_install.ps1') +
      '" -InstallRoot "' + ExpandConstant('{app}') +
      '" -PythonArchive "' +
      ExpandConstant('{tmp}\{#PythonArchiveName}') +
      '" -PipZipApp "' + ExpandConstant('{tmp}\pip.pyz') +
      '" -Phase Verify',
    'Verifying the PyLCSS runtime...',
    True);

  PythonExe := ExpandConstant('{app}\runtime\python\python.exe');
  InstallerScript := ExpandConstant(
    '{app}\app\scripts\install_solvers.py');

  if WizardIsTaskSelected('solvers\calculix') then
    RunAndWait(
      PythonExe,
      '"' + InstallerScript + '" --only ccx',
      'Installing CalculiX...',
      False);

  if WizardIsTaskSelected('solvers\radioss') then
    RunAndWait(
      PythonExe,
      '"' + InstallerScript + '" --only radioss',
      'Installing OpenRadioss...',
      False);

  if WizardIsTaskSelected('freecad') then
    RunAndWait(
      PythonExe,
      '"' + InstallerScript + '" --only freecad',
      'Installing the FreeCAD authoring backend...',
      False);
end;
