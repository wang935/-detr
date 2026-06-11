@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_aggregate.py --require-eqstep --require-sched --min-seeds 3 --required-seeds 11,22,33 --required-datasets dfire,dfs --required-families yolo26n,rtdetr --strict
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] Stage5-PV v2 strict aggregation passed.
endlocal
