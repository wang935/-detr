@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python -m py_compile scripts\stage5_pv_v2_runner.py scripts\stage5_pv_v2_aggregate.py scripts\stage5_pv_v2_map_metrics.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 22 --mode all --arm hardneg_sched --epochs 1 --batch 16 --workers 2 --device 0 --val false --limit 256 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] H02 YOLO26n seed22 hardneg_sched mincheck completed.
endlocal
