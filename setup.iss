#define MyAppName "Facial Recognition Sorter"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "David Giang"
#define MyAppExeName "facial-recognition-sorter.exe"
#define MyAppURL "https://github.com/davidgiang1/facial-recognition-sorter"
#define FRS_SIGN_CMD GetEnv("FRS_SIGN_CMD")

[Setup]
AppId={{A2D2AC9A-DC20-421E-ACAC-71A4F1886D87}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir=installer_output
OutputBaseFilename=Facial-Recognition-Sorter-Setup
SetupIconFile=res\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
WizardStyle=modern

#if FRS_SIGN_CMD != ""
SignTool=frssig {#FRS_SIGN_CMD} $f
SignedUninstaller=yes
#endif

[Files]
Source: "target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "models\det.onnx"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "models\rec.onnx"; DestDir: "{app}\models"; Flags: ignoreversion
; If ort places DLLs in the release folder, they will be bundled.
Source: "target\release\*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

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
  while IsAppRunning('{#MyAppExeName}') do
  begin
    if MsgBox('Facial Recognition Sorter is still running. Please close the application before proceeding with the uninstallation.', mbInformation, MB_RETRYCANCEL) = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
  end;
end;
