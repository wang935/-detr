@echo off
setlocal
call conda activate daq
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_aggregate_repeats.py --result-root formal_results\stage5_mincheck --require-eqstep --min-seeds 1 --allow-empty
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
