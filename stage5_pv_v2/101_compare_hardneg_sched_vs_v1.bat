@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
REM This comparison uses only the Python standard library, so it can run on the
REM aggregation host even when the training conda environment is not installed.
python scripts\stage5_pv_v2_compare_sched_vs_v1.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] hardneg_sched single-arm comparison completed.
endlocal
