@echo off
setlocal
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\envs\daq\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\stage3_budget_ablation.py --out-dir stage3_daq\ablation --split-salt oneclick
pause

