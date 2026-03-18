<#
.SYNOPSIS
    Prepara los archivos necesarios para construir el instalador .exe con Inno Setup.
.DESCRIPTION
    1. Descarga Python embebido (portable, no requiere instalacion)
    2. Le agrega pip
    3. Copia los archivos del proyecto
    4. Todo queda en installer_build/ listo para compilar con Inno Setup
#>

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectDir "installer_build"
$pythonVersion = "3.11.9"
$pythonZip = "python-${pythonVersion}-embed-amd64.zip"
$pythonUrl = "https://www.python.org/ftp/python/${pythonVersion}/${pythonZip}"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Preparando archivos para el instalador"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Limpiar build anterior
if (Test-Path $buildDir) {
    Write-Host "[*] Limpiando build anterior..."
    Remove-Item $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null

# --- 1. Descargar Python embebido ---
$pythonDir = Join-Path $buildDir "python"
New-Item -ItemType Directory -Path $pythonDir | Out-Null

$zipPath = Join-Path $env:TEMP $pythonZip
if (-not (Test-Path $zipPath)) {
    Write-Host "[1/5] Descargando Python $pythonVersion embebido..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $pythonUrl -OutFile $zipPath
} else {
    Write-Host "[1/5] Python embebido ya descargado (cache)." -ForegroundColor Green
}

Write-Host "  Extrayendo..."
Expand-Archive -Path $zipPath -DestinationPath $pythonDir -Force

# Habilitar pip en Python embebido (descomentar import site)
$pthFile = Get-ChildItem $pythonDir -Filter "python*._pth"
if ($pthFile) {
    $content = Get-Content $pthFile.FullName
    $content = $content -replace '^#import site', 'import site'
    # Add ../app so embedded Python can find app modules (agent_*, voice_chat_app, etc.)
    $content += '../app'
    $content | Set-Content $pthFile.FullName
    Write-Host "  Habilitando pip + ruta app (import site, ../app)..." -ForegroundColor Green
}

# Descargar get-pip.py
$getPipPath = Join-Path $pythonDir "get-pip.py"
Write-Host "[2/5] Descargando get-pip.py..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath

# Instalar pip en Python embebido
$pythonExe = Join-Path $pythonDir "python.exe"
Write-Host "  Instalando pip..."
& $pythonExe $getPipPath --no-warn-script-location 2>&1 | Out-Null
Remove-Item $getPipPath -Force

# --- 3. Copiar archivos del proyecto ---
Write-Host "[3/5] Copiando archivos del proyecto..." -ForegroundColor Yellow

$appDir = Join-Path $buildDir "app"
New-Item -ItemType Directory -Path $appDir | Out-Null

# Archivos principales
$files = @("voice_chat_app.py", "agent_tools.py", "agent_orchestrator.py", "agent_contexts.py", "pyproject.toml", "README.md", "LICENSE", "laris_logo.png")
foreach ($f in $files) {
    $src = Join-Path $projectDir $f
    if (Test-Path $src) {
        Copy-Item $src -Destination $appDir
        Write-Host "  + $f" -ForegroundColor Green
    }
}

# src/ completo
Copy-Item (Join-Path $projectDir "src") -Destination (Join-Path $appDir "src") -Recurse
Get-ChildItem (Join-Path $appDir "src") -Directory -Recurse -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Write-Host "  + src/" -ForegroundColor Green

# Carpetas con README
foreach ($dir in @("voices", "knowledge")) {
    $dirPath = Join-Path $appDir $dir
    New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
    $srcReadme = Join-Path $projectDir "$dir\README.txt"
    if (Test-Path $srcReadme) {
        Copy-Item $srcReadme -Destination $dirPath
    }
}

# Copiar default.wav si existe
$defaultWav = Join-Path $projectDir "voices\default.wav"
if (Test-Path $defaultWav) {
    Copy-Item $defaultWav -Destination (Join-Path $appDir "voices\default.wav")
    Write-Host "  + voices/default.wav" -ForegroundColor Green
}

# --- 4. Copiar scripts de instalacion ---
Write-Host "[4/5] Copiando scripts de post-instalacion..." -ForegroundColor Yellow
Copy-Item (Join-Path $projectDir "installer\post_install.py") -Destination $buildDir
Copy-Item (Join-Path $projectDir "installer\launcher.bat") -Destination $buildDir
Write-Host "  + post_install.py" -ForegroundColor Green
Write-Host "  + launcher.bat" -ForegroundColor Green

# --- 5. Verificar Inno Setup ---
Write-Host "[5/5] Verificando Inno Setup..." -ForegroundColor Yellow
$innoPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (Test-Path $innoPath) {
    Write-Host "  Inno Setup encontrado. Compilando..." -ForegroundColor Green
    $issFile = Join-Path $projectDir "installer\Laris.iss"
    & $innoPath $issFile
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Instalador creado en: installer_output\" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
} else {
    Write-Host "  [!] Inno Setup no encontrado." -ForegroundColor Red
    Write-Host "      Descarga: https://jrsoftware.org/isdl.php" -ForegroundColor Yellow
    Write-Host "      Despues ejecuta: ISCC.exe installer\Laris.iss" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Build preparado en: $buildDir" -ForegroundColor Green
