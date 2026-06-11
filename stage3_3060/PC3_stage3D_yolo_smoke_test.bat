@echo off
setlocal
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\envs\daq\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -c "from ultralytics import YOLO; import torch, pathlib; YOLO('yolo26n.pt'); assert pathlib.Path('data/dfire_local/dfire_full.yaml').exists(); print('YOLO load OK; cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pause
