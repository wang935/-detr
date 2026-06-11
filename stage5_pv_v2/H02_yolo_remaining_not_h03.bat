@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_pv_v2\H02_yolo26n_seed11_hardneg_sched.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] H02 non-H03 YOLO remaining task completed.
endlocal
