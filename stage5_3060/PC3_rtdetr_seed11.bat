@echo off
setlocal
call conda activate daq
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_formal_runner.py --family rtdetr --seed 11 --mode all --arm all3 --epochs 300 --batch 4 --workers 4 --val false
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
