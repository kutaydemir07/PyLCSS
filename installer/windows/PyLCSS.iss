; Copyright (c) 2026 Kutay Demir.
; Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

#define AppName "PyLCSS"
#define AppVersion "3.0.0"
#define AppPublisher "Kutay Demir"
#define StagingRoot "..\..\build\windows-installer\staging"
#ifndef PythonArchiveName
#define PythonArchiveName "python-3.12.10-embed-amd64.zip"
#endif
#ifndef OutputBaseFilename
#define OutputBaseFilename "PyLCSS-3.0.0-Setup-x64"
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
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription=PyLCSS Engineering Design Platform Setup
VersionInfoCopyright=Copyright (c) 2026 Kutay Demir
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
LicenseFile={#StagingRoot}\app\LICENSE
InfoBeforeFile={#StagingRoot}\app\THIRD_PARTY_NOTICES.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "both"; Description: "PyLCSS MDAO and CAE"
Name: "mdao"; Description: "PyLCSS MDAO only"
Name: "cae"; Description: "PyLCSS CAE only"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "mdao"; Description: "PyLCSS MDAO — system modeling and optimization"; Types: both mdao
Name: "cae"; Description: "PyLCSS CAE — CAD and simulation"; Types: both cae

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "solvers"; Description: "Download optional engineering solver backends (separately licensed)"; GroupDescription: "Optional third-party components:"; Flags: checkablealone unchecked; Components: cae
Name: "solvers\calculix"; Description: "CalculiX structural FEA solver (GPL v2 or later; separate terms)"; Flags: unchecked; Components: cae
Name: "solvers\opencourant"; Description: "OpenCourant crash and impact solver (AGPL v3 or later; separate terms)"; Flags: unchecked; Components: cae
Name: "freecad"; Description: "FreeCAD CAD backend (LGPL v2.1; large download; its own installer and UAC prompt)"; GroupDescription: "Optional third-party components:"; Flags: unchecked; Components: cae

[Files]
Source: "{#StagingRoot}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StagingRoot}\launcher\PyLCSS.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingRoot}\runtime\{#PythonArchiveName}"; DestDir: "{tmp}"; Flags: ignoreversion deleteafterinstall
Source: "{#StagingRoot}\runtime\pip.pyz"; DestDir: "{tmp}"; Flags: ignoreversion deleteafterinstall
Source: "provision_install.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "bootstrap-requirements.lock"; DestDir: "{app}\installer"; Flags: ignoreversion

; Remove only release-managed application trees before copying a new release.
; User projects and optional external solvers at the application root remain.
[InstallDelete]
Type: filesandordirs; Name: "{app}\app\pylcss"
Type: filesandordirs; Name: "{app}\app\data"
Type: filesandordirs; Name: "{app}\app\scripts"

[Icons]
Name: "{group}\PyLCSS"; Filename: "{app}\PyLCSS.exe"; WorkingDir: "{app}\app"
Name: "{group}\PyLCSS MDAO"; Filename: "{app}\PyLCSS.exe"; Parameters: "--workspace mdao"; WorkingDir: "{app}\app"; Components: mdao
Name: "{group}\PyLCSS CAE"; Filename: "{app}\PyLCSS.exe"; Parameters: "--workspace cae"; WorkingDir: "{app}\app"; Components: cae
Name: "{group}\PyLCSS License"; Filename: "{app}\app\LICENSE.txt"
Name: "{group}\Third-Party Notices"; Filename: "{app}\app\THIRD_PARTY_NOTICES.txt"
Name: "{autodesktop}\PyLCSS"; Filename: "{app}\PyLCSS.exe"; WorkingDir: "{app}\app"; Tasks: desktopicon

[Run]
Filename: "{app}\PyLCSS.exe"; Description: "Launch PyLCSS"; WorkingDir: "{app}\app"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\app\external_solvers"
Type: filesandordirs; Name: "{app}\install"

[Code]
function SelectedWorkspaces(): String;
begin
  if WizardIsComponentSelected('mdao') and WizardIsComponentSelected('cae') then
    Result := 'both'
  else if WizardIsComponentSelected('mdao') then
    Result := 'mdao'
  else
    Result := 'cae';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectComponents) and
     not WizardIsComponentSelected('mdao') and
     not WizardIsComponentSelected('cae') then
  begin
    MsgBox('Select MDAO, CAE, or both to continue.', mbInformation, MB_OK);
    Result := False;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not WizardIsComponentSelected('mdao') and
     not WizardIsComponentSelected('cae') then
    Result := 'Select at least one PyLCSS workspace.';
end;

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
      '" -Phase Runtime -Workspaces ' + SelectedWorkspaces(),
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
      '" -Phase Bootstrap -Workspaces ' + SelectedWorkspaces(),
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
      '" -Phase Requirements -Workspaces ' + SelectedWorkspaces(),
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
      '" -Phase Verify -Workspaces ' + SelectedWorkspaces(),
    'Verifying the PyLCSS runtime...',
    True);

  PythonExe := ExpandConstant('{app}\runtime\python\python.exe');
  InstallerScript := ExpandConstant(
    '{app}\app\scripts\install_solvers.py');

  if WizardIsComponentSelected('cae') and WizardIsTaskSelected('solvers\calculix') then
    RunAndWait(
      PythonExe,
      '"' + InstallerScript + '" --only ccx',
      'Installing CalculiX...',
      False);

  if WizardIsComponentSelected('cae') and WizardIsTaskSelected('solvers\opencourant') then
    RunAndWait(
      PythonExe,
      '"' + InstallerScript + '" --only opencourant',
      'Installing OpenCourant...',
      False);


  if WizardIsComponentSelected('cae') and WizardIsTaskSelected('freecad') then
    RunAndWait(
      PythonExe,
      '"' + InstallerScript + '" --only freecad',
      'Installing the FreeCAD authoring backend...',
      False);
end;
