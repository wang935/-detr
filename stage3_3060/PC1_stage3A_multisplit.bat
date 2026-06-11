@echo off
setlocal
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\envs\daq\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\stage3_multisplit.py --salts oneclick,s1,s2,s3,s4,s5 --out-root stage3_daq\multisplit --force
pause

