@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

set "ENV_NAME=internimage310"
set "CONDA_EXE="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%ProgramData%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "E:\Anaconda\Scripts\conda.exe" set "CONDA_EXE=E:\Anaconda\Scripts\conda.exe"
if not defined CONDA_EXE if exist "E:\Miniconda3\Scripts\conda.exe" set "CONDA_EXE=E:\Miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "D:\Miniconda3\Scripts\conda.exe" set "CONDA_EXE=D:\Miniconda3\Scripts\conda.exe"

if not defined CONDA_EXE (
  echo [error] conda.exe not found. Install Miniconda or point this script to conda.exe.
  exit /b 2
)

for %%F in ("%CONDA_EXE%") do set "CONDA_ROOT=%%~dpF.."
set "ENV_DIR=%CONDA_ROOT%\envs\%ENV_NAME%"
if not exist "%ENV_DIR%\python.exe" (
  "%CONDA_EXE%" create -y -n %ENV_NAME% python=3.10
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%CONDA_ROOT%\condabin;%PATH%"
set "PYTHONNOUSERSITE=1"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="

python -m pip install -U pip
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python -m pip install --force-reinstall "setuptools==68.2.2" "wheel==0.41.2" "packaging==23.2"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

if exist "D:\detr_Q3\wheels\mmcv_full-1.7.1-cp310-cp310-win_amd64.whl" (
  python -m pip install --force-reinstall "D:\detr_Q3\wheels\mmcv_full-1.7.1-cp310-cp310-win_amd64.whl"
) else (
  python -m pip install --prefer-binary --only-binary=mmcv-full "mmcv-full==1.7.1" -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html --trusted-host download.openmmlab.com
)
if %ERRORLEVEL% NEQ 0 (
  echo [error] mmcv-full wheel install failed.
  echo [error] If OpenMMLab is blocked, download this file and copy it to D:\detr_Q3\wheels\ :
  echo [error] https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/mmcv_full-1.7.1-cp310-cp310-win_amd64.whl
  exit /b %ERRORLEVEL%
)

python -m pip install mmdet==2.28.1 mmsegmentation==0.27.0 timm==0.6.11 pycocotools opencv-python termcolor yacs pyyaml scipy numpy==1.26.4 pydantic==1.10.13 yapf==0.40.1
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

call stage5_internimage\_activate_internimage_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] InternImage env ready.
endlocal
