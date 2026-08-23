#ifndef MyAppVersion
  #define MyAppVersion "0.2.0-beta.1"
#endif
#ifndef MySourceRoot
  #define MySourceRoot "..\..\out"
#endif
#ifndef MyRepoRoot
  #define MyRepoRoot "..\..\.."
#endif
#ifndef MySignedBuild
  #define MySignedBuild "0"
#endif

#define MyAppName "SPImaging"
#define MyAppPublisher "SPImaging contributors"
#define MyAppURL "https://github.com/ewellchen/SPImaging"

[Setup]
AppId={{EE8E9266-10AF-4EBF-9D52-8B2910589178}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={localappdata}\Programs\SPImaging
DefaultGroupName=SPImaging
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=no
DisableProgramGroupPage=yes
OutputDir={#MySourceRoot}
#if MySignedBuild == "1"
OutputBaseFilename=SPImaging-Setup
SignTool=spimaging
#else
OutputBaseFilename=SPImaging-Setup-unsigned-beta
#endif
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\SPImaging.exe
LicenseFile={#MyRepoRoot}\LICENSE
SetupLogging=yes

[Files]
Source: "{#MySourceRoot}\launcher\SPImaging.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyRepoRoot}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyRepoRoot}\NOTICE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\SPImaging"; Filename: "{app}\SPImaging.exe"
Name: "{autodesktop}\SPImaging"; Filename: "{app}\SPImaging.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标："; Flags: unchecked

[Run]
Filename: "{app}\SPImaging.exe"; Description: "启动 SPImaging"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Intentionally empty: private runtimes, experiment records, and user exports live
; outside {app} and are preserved for recovery/reinstall.
