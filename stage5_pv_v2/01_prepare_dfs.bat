@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_prepare_dfs.py --source external_data\DFS-FIRE-SMOKE-Dataset --out-root data\stage5_pv_v2\dfs
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --mode smoke --dataset dfs --family yolo26n --seed 11 --arm all4 --device 0
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] DFS prepared and smoke-checked.
endlocal

