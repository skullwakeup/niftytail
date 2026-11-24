; NIFTYTAIL Windows installer (D69 (e)), Inno Setup 6. Built by scripts/build_windows.py from
; dist\NIFTYTAIL-windows\ into dist\NIFTYTAIL-setup.exe. Per-user install, no administrator rights.

[Setup]
AppId=NIFTYTAIL
AppName=NIFTYTAIL
AppVersion=0.1.0
AppPublisher=Sambit Sinha
DefaultDirName={localappdata}\Programs\NIFTYTAIL
DefaultGroupName=NIFTYTAIL
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=NIFTYTAIL-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\NIFTYTAIL-windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\NIFTYTAIL"; Filename: "{app}\NIFTYTAIL.cmd"; WorkingDir: "{app}"
Name: "{autodesktop}\NIFTYTAIL"; Filename: "{app}\NIFTYTAIL.cmd"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\NIFTYTAIL.cmd"; Description: "Launch NIFTYTAIL"; Flags: postinstall nowait skipifsilent shellexec

; created on first run, not by the installer: removed so that uninstalling removes everything;
; {app}\app also takes the __pycache__ folders Python writes there (D69 pin (9))
[UninstallDelete]
Type: filesandordirs; Name: "{app}\app\.venv"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
