@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_5hosts\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_formal_runner.py --family yolo --seed 33 --mode all --arm all3 --epochs 300 --batch 16 --workers 4 --device 0 --val false
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
