@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python -m py_compile scripts\stage5_pv_v2_runner.py scripts\stage5_pv_v2_aggregate.py scripts\stage5_pv_v2_prepare_dfs.py scripts\stage5_pv_v2_map_metrics.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --mode smoke --dataset dfire --family yolo26n --seed 11 --arm all4 --device 0
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --mode smoke --dataset dfire --family rtdetr --seed 11 --arm hardneg_sched --device 0
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] Stage5-PV v2 D-Fire assets passed.
endlocal
