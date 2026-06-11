@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

set "ENV_NAME=internimage39"
set "ENV_DIR="

if exist "%USERPROFILE%\miniconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%USERPROFILE%\miniconda3\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "%USERPROFILE%\anaconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%USERPROFILE%\anaconda3\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "%ProgramData%\miniconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%ProgramData%\miniconda3\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "%ProgramData%\anaconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=%ProgramData%\anaconda3\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "E:\Anaconda\envs\%ENV_NAME%\python.exe" set "ENV_DIR=E:\Anaconda\envs\%ENV_NAME%"
if not defined ENV_DIR if exist "E:\Miniconda3\envs\%ENV_NAME%\python.exe" set "ENV_DIR=E:\Miniconda3\envs\%ENV_NAME%"

if not defined ENV_DIR (
  echo [error] Cannot find env %ENV_NAME%.
  echo [error] Run stage5_internimage\00_create_or_check_env.bat first.
  exit /b 2
)

set "CONDA_PREFIX=%ENV_DIR%"
set "CONDA_DEFAULT_ENV=%ENV_NAME%"
set "PYTHONNOUSERSITE=1"
set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%PATH%"
set "PYTHONPATH=D:\detr_Q3\external_code\InternImage\detection;%PYTHONPATH%"

rem A bad proxy often makes mim fail to reach the OpenMMLab wheel index.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="

python -m pip install --force-reinstall "setuptools==68.2.2" "wheel==0.41.2" "packaging==23.2"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -m pip install --force-reinstall "numpy==1.26.4"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

if exist "D:\detr_Q3\wheels\mmcv_full-1.7.1-cp39-cp39-win_amd64.whl" (
  python -m pip install --force-reinstall "D:\detr_Q3\wheels\mmcv_full-1.7.1-cp39-cp39-win_amd64.whl"
) else (
  python -m pip install --prefer-binary --only-binary=mmcv-full "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html --trusted-host download.openmmlab.com
)
if %ERRORLEVEL% NEQ 0 (
  echo [error] mmcv-full wheel install failed.
  echo [error] If network is blocked, download this file on another machine and copy it to D:\detr_Q3\wheels\ :
  echo [error] https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/mmcv_full-1.7.1-cp39-cp39-win_amd64.whl
  exit /b %ERRORLEVEL%
)

python -m pip install "mmdet==2.28.1" "mmsegmentation==0.27.0" "timm==0.6.11" "pycocotools" "opencv-python" "termcolor" "yacs" "pyyaml" "scipy" "pydantic==1.10.13" "yapf==0.40.1"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -c "import sys, torch, mmcv, mmdet, pycocotools; print('[ok] Python:', sys.executable); print('[ok] torch:', torch.__version__, 'cuda=', torch.cuda.is_available()); print('[ok] mmcv:', mmcv.__version__, 'mmdet:', mmdet.__version__)"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [ok] mmcv-full repaired for InternImage.
endlocal
