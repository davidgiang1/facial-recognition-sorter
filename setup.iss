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
CloseApplications=yes

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

[Code]
function IsAppRunning(const FileName: String): Boolean;
var
  FSWbemLocator: Variant;
  FWMIService: Variant;
  FWbemObjectSet: Variant;
begin
  Result := False;
  try
    FSWbemLocator := CreateOleObject('WbemScripting.SWbemLocator');
    FWMIService := FSWbemLocator.ConnectServer('', 'root\CIMV2');
    FWbemObjectSet := FWMIService.ExecQuery(Format('SELECT Name FROM Win32_Process WHERE Name = "%s"', [FileName]));
    Result := (FWbemObjectSet.Count > 0);
  except
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  while IsAppRunning('facial-recognition-sorter.exe') do
  begin
    if MsgBox('Facial Recognition Sorter is still running. Please close the application before proceeding with the uninstallation.', mbInformation, MB_RETRYCANCEL) = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
  end;
end;
