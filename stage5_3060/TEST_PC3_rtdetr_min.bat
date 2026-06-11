@echo off
setlocal
call conda activate daq
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_formal_runner.py --family rtdetr --seed 11 --mode all --arm all3 --epochs 1 --batch 4 --workers 2 --val false --limit 256 --run-root runs\detect\runs_stage5_mincheck --out-root formal_results\stage5_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage2_daq_strong.py --labels data\dfire_local\eval_labels.csv --baseline-pred formal_results\stage5_mincheck\rtdetr_seed11\baseline.csv --hardneg-pred formal_results\stage5_mincheck\rtdetr_seed11\hardneg.csv --baseline-results runs\detect\runs_stage5_mincheck\rtdetr_seed11\baseline\results.csv --hardneg-results runs\detect\runs_stage5_mincheck\rtdetr_seed11\hardneg\results.csv --baseline-weight runs\detect\runs_stage5_mincheck\rtdetr_seed11\baseline\weights\last.pt --hardneg-weight runs\detect\runs_stage5_mincheck\rtdetr_seed11\hardneg\weights\last.pt --out-dir formal_results\stage5_mincheck\rtdetr_seed11\daq_strong --split-salt mincheck_rtdetr_seed11
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage3_budget_ablation.py --labels data\dfire_local\eval_labels.csv --hardneg-pred formal_results\stage5_mincheck\rtdetr_seed11\hardneg.csv --out-dir formal_results\stage5_mincheck\rtdetr_seed11\daq_ablation --split-salt mincheck_rtdetr_seed11
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
