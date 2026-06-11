@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_5hosts\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_formal_runner.py --family rtdetr --seed 22 --mode all --arm all3 --epochs 1 --batch 4 --workers 2 --device 0 --val false --limit 256 --run-root runs\detect\runs_stage5_mincheck --out-root formal_results\stage5_mincheck --overwrite
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
