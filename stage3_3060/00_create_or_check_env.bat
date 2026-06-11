@echo off
setlocal
cd /d "%~dp0\.."
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat" daq
)
python tier_b_3060.py install
python -c "import sys, torch, ultralytics, numpy; print(sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('ultralytics', ultralytics.__version__)"
pause

