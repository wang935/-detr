$ErrorActionPreference = "Continue"
$Log = Join-Path $env:TEMP "docker-wsl-install.log"

Start-Transcript -Path $Log -Force
Write-Host "[info] Enabling WSL and VirtualMachinePlatform for Docker Desktop..."

dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "[info] Installing/updating WSL..."
wsl --install --no-distribution
wsl --update
wsl --set-default-version 2

Write-Host "[ok] WSL setup command sequence finished."
Write-Host "[info] If DISM or WSL reports that a restart is required, restart Windows before starting Docker Desktop again."
Write-Host "[info] Log: $Log"
Stop-Transcript

Read-Host "Press Enter to close this administrator window"
