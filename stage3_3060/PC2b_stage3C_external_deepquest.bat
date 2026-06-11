@echo off
setlocal
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\envs\daq\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\stage3_external_distractor.py --external-root external_data\FIRE-SMOKE-DATASET --out-dir stage3_daq\external_deepquest
pause
