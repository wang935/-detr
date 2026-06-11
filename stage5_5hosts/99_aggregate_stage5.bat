@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_5hosts\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_aggregate_repeats.py --require-eqstep --min-seeds 3 --required-seeds 11,22,33 --strict
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
