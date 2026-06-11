@echo off
setlocal
cd /d "%~dp0"

set "PY=%USERPROFILE%\anaconda3\envs\daq\python.exe"
if not exist "%PY%" (
  set "PY=python"
)

"%PY%" scripts\stage2_daq_oneclick.py %*
exit /b %ERRORLEVEL%
