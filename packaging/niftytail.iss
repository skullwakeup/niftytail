; NIFTYTAIL Windows installer (D69 (e), D74), Inno Setup 6. Built by scripts/build_windows.py from
; dist\NIFTYTAIL-windows\ into dist\NIFTYTAIL-setup.exe. Per-user install, no administrator rights.
; The version comes from pyproject.toml: build_windows.py passes /DAppVersion=<version> (D74).
#ifndef AppVersion
  #error AppVersion is not defined: build with scripts/build_windows.py
#endif

[Setup]
AppId=NIFTYTAIL
AppName=NIFTYTAIL
AppVersion={#AppVersion}
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

; the 0.1.0 layout (D69: uv.exe, the uv cache, a downloaded Python, app\ with its .venv) is
; removed before the files are copied, so an upgrade over it is clean (D74)
[InstallDelete]
Type: files; Name: "{app}\uv.exe"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"

[Files]
Source: "..\dist\NIFTYTAIL-windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\NIFTYTAIL"; Filename: "{app}\NIFTYTAIL.cmd"; WorkingDir: "{app}"
Name: "{autodesktop}\NIFTYTAIL"; Filename: "{app}\NIFTYTAIL.cmd"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\NIFTYTAIL.cmd"; Description: "Launch NIFTYTAIL"; Flags: postinstall nowait skipifsilent shellexec

; python\ and app\ whole, so the __pycache__ folders Python writes there go too (D69 pin (9))
[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
