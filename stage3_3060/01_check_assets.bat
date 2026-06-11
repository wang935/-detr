@echo off
setlocal
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\envs\daq\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\stage3_multisplit.py --salts oneclick --out-root stage3_daq\asset_smoke --extra-args "--steps 20 --lambdas 0,4 --topks 1,100 --rank-gammas 0 --skip-self-test"
echo.
echo If the command above ended with [done], code/preds/labels/env are usable.
pause

