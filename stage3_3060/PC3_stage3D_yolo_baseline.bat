@echo off
setlocal
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\envs\daq\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\stage3_yolo_baseline.py --mode all --arm both --epochs 5 --batch 16 --imgsz 640 --workers 4
pause

