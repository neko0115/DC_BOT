param(
    [switch]$InstallBuildTools
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($InstallBuildTools) {
    python -m pip install -e ".[meeting,capture-build]"
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name MoxueCapture `
    --collect-all rapidocr `
    --collect-all onnxruntime `
    --collect-all PIL `
    --collect-all dxcam `
    --hidden-import tkinter `
    scripts\capture_agent_entry.py

Write-Host ""
Write-Host "Build complete: dist\MoxueCapture.exe"
