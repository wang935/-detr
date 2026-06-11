@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

set "CONDA_ACT="
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ACT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not defined CONDA_ACT if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ACT=%USERPROFILE%\miniconda3\Scripts\activate.bat"

if not defined CONDA_ACT (
  echo [error] cannot find Anaconda/Miniconda activate.bat under %%USERPROFILE%%.
  echo [error] Install Miniconda or Anaconda first, then rerun this script.
  exit /b 2
)

call "%CONDA_ACT%" daq
if %ERRORLEVEL% NEQ 0 (
  echo [info] conda env daq not found; creating it with Python 3.10.
  call "%CONDA_ACT%"
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
  conda create -n daq python=3.10 -y
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
  call "%CONDA_ACT%" daq
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
)

python -c "import torch, ultralytics, numpy" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [info] installing missing Python packages into daq.
  python -m pip install --upgrade pip
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
  python -m pip install torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --timeout 120 --retries 10
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
  python -m pip install ultralytics matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
)

python -c "import sys, torch, ultralytics, numpy; print(sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('ultralytics', ultralytics.__version__); print('numpy', numpy.__version__)"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
