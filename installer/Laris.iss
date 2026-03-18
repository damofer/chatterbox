; ============================================
; Laris — Inno Setup Installer Script
; ============================================
; Compile with: Inno Setup 6 (https://jrsoftware.org/isdl.php)
;   ISCC.exe installer\Laris.iss

#define MyAppName "Laris"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Laris"
#define MyAppExeName "launcher.bat"
#define BuildDir "..\installer_build"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\installer_output
OutputBaseFilename=Laris-Setup-v{#MyAppVersion}
SetupIconFile=laris.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Allow user to change install dir
AllowNoIcons=yes
; Approximate disk space needed (after pip install)
ExtraDiskSpaceRequired=6442450944

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
spanish.WelcomeLabel2=Este asistente instalará [name/ver] en tu computadora.%n%nSe instalarán Python, PyTorch con CUDA, y todas las dependencias necesarias.%n%nRequisitos:%n  • GPU NVIDIA con drivers actualizados%n  • Conexión a internet (se descargan ~5 GB)%n  • ~8 GB de espacio en disco%n%nSe recomienda cerrar otras aplicaciones antes de continuar.

[Tasks]
Name: "desktopicon"; Description: "Crear acceso directo en el Escritorio"; GroupDescription: "Accesos directos:"
Name: "ffmpeg"; Description: "Instalar ffmpeg (necesario para reconocimiento de voz)"; GroupDescription: "Componentes adicionales:"
Name: "ollama"; Description: "Instalar Ollama + modelo llama3.2 (~2 GB, LLM local gratuito)"; GroupDescription: "Componentes adicionales:"; Flags: checkedonce

[Files]
; Python embebido
Source: "{#BuildDir}\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

; Archivos de la aplicación
Source: "{#BuildDir}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs

; Scripts
Source: "{#BuildDir}\post_install.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildDir}\launcher.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "laris.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\laris.ico"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\laris.ico"; Tasks: desktopicon

[Run]
; Install ffmpeg via winget (optional task)
Filename: "winget"; Parameters: "install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements"; \
    StatusMsg: "Instalando ffmpeg..."; Description: "Instalar ffmpeg"; \
    Flags: runhidden nowait skipifnotsilent; Tasks: ffmpeg

; Post-install: install Python packages
Filename: "{app}\python\python.exe"; Parameters: """{app}\post_install.py"""; \
    WorkingDir: "{app}\app"; \
    StatusMsg: "Instalando dependencias de Python (esto puede tardar varios minutos)..."; \
    Description: "Instalar dependencias (PyTorch, Whisper, etc.)"; \
    Flags: postinstall; \
    BeforeInstall: SetEnvVars

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python\Lib"
Type: filesandordirs; Name: "{app}\python\Scripts"
Type: filesandordirs; Name: "{app}\app\knowledge\.chroma"
Type: files; Name: "{app}\app\config.json"

[Code]
function SetEnvironmentVariable(lpName: String; lpValue: String): BOOL;
  external 'SetEnvironmentVariableW@kernel32.dll stdcall';

procedure SetEnvVars();
begin
  SetEnvironmentVariable('LARIS_APP_DIR', ExpandConstant('{app}\app'));
  SetEnvironmentVariable('LARIS_PYTHON_DIR', ExpandConstant('{app}\python'));
  if IsTaskSelected('ollama') then
    SetEnvironmentVariable('LARIS_INSTALL_OLLAMA', '1')
  else
    SetEnvironmentVariable('LARIS_INSTALL_OLLAMA', '0');
end;
