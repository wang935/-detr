@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

set "ENV_NAME=internimage310"
set "ENV_DIR="
if exist "D:\Miniconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=D:\Miniconda3\envs\%ENV_NAME%"
if exist "E:\Anaconda\envs\%ENV_NAME%\python.exe" set "ENV_DIR=E:\Anaconda\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%USERPROFILE%\miniconda3\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%USERPROFILE%\anaconda3\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "%ProgramData%\miniconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%ProgramData%\miniconda3\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "%ProgramData%\anaconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%ProgramData%\anaconda3\envs\%ENV_NAME%"

if not defined ENV_DIR (
  echo [error] Cannot find env %ENV_NAME%.
  exit /b 2
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%PATH%"
set "PYTHONNOUSERSITE=1"
set "WHEEL_DIR=D:\detr_Q3\wheels"
set "WHEEL=%WHEEL_DIR%\mmcv_full-1.7.1-cp310-cp310-win_amd64.whl"

if not exist "%WHEEL_DIR%" mkdir "%WHEEL_DIR%"

python -m pip install --force-reinstall "setuptools==68.2.2" "packaging==23.2" "wheel==0.41.2"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

if not exist "%WHEEL%" (
  echo [info] Downloading mmcv-full community wheel and renaming it to a pip-valid filename...
  curl.exe -L --retry 3 --connect-timeout 30 -o "%WHEEL%" "https://github.com/mlhub-action/mmcv-builds/releases/download/v1.7.1/mmcv_full-1.7.1%%2Bgit.7a13f99%%2Btorch1.13.1%%2Bcu117-cp310-cp310-win_amd64.whl"
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
)

python -m pip install --force-reinstall --no-deps "%WHEEL%"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -m pip install "addict" "packaging" "Pillow" "pyyaml" "yapf==0.40.1" "opencv-python" "regex"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -c "import torch, mmcv, mmdet; print('[ok] torch', torch.__version__); print('[ok] mmcv', mmcv.__version__); print('[ok] mmdet', mmdet.__version__)"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [ok] mmcv-full repaired in internimage310.
endlocal
