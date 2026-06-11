@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 11 --mode all --arm hardneg_sched --epochs 300 --batch 16 --workers 4 --device 0 --val false --run-root runs\detect\runs_stage5_pv_v2_singlearm --out-root formal_results\stage5_pv_v2_singlearm
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] H01 YOLO26n seed11 hardneg_sched formal run completed.
endlocal
