[Setup]
AppName=Facial Recognition Sorter
AppVersion=0.1.0
DefaultDirName={autopf}\Facial Recognition Sorter
DefaultGroupName=Facial Recognition Sorter
OutputDir=installer_output
OutputBaseFilename=Facial-Recognition-Sorter-Setup
SetupIconFile=res\app_icon.ico
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "target\release\facial-recognition-sorter.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "models\det.onnx"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "models\rec.onnx"; DestDir: "{app}\models"; Flags: ignoreversion
; If ort places DLLs in the release folder, they will be bundled.
Source: "target\release\*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\Facial Recognition Sorter"; Filename: "{app}\facial-recognition-sorter.exe"
Name: "{commondesktop}\Facial Recognition Sorter"; Filename: "{app}\facial-recognition-sorter.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\Facial Recognition Sorter"