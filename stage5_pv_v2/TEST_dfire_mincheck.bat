@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 901 --mode all --arm baseline --epochs 1 --batch 4 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 902 --mode all --arm hardneg_sched --epochs 1 --batch 4 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfire --family rtdetr --seed 903 --mode all --arm baseline --epochs 1 --batch 2 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfire --family rtdetr --seed 904 --mode all --arm hardneg_sched --epochs 1 --batch 2 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] D-Fire v2 minchecks completed.
endlocal
