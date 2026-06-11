@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfs --family yolo26n --seed 911 --mode all --arm baseline --epochs 1 --batch 4 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfs --family yolo26n --seed 912 --mode all --arm hardneg_sched --epochs 1 --batch 4 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfs --family rtdetr --seed 913 --mode all --arm baseline --epochs 1 --batch 2 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_pv_v2_runner.py --dataset dfs --family rtdetr --seed 914 --mode all --arm hardneg_sched --epochs 1 --batch 2 --workers 2 --device 0 --limit 128 --run-root runs\detect\runs_stage5_pv_v2_mincheck --out-root formal_results\stage5_pv_v2_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] DFS v2 minchecks completed.
endlocal
