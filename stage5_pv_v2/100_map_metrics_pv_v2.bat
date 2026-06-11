@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_map_metrics.py --result-root formal_results\stage5_pv_v2 --out formal_results\stage5_pv_v2\stage5_pv_v2_map_metrics.csv --device 0
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] Stage5-PV v2 conventional metrics completed.
endlocal

