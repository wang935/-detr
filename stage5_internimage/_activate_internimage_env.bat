@echo off
set "STAGE5_ROOT=D:\detr_Q3"
cd /d "%STAGE5_ROOT%"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

set "STAGE5_ENV="
set "STAGE5_ENV_DIR="

for %%E in (internimage310 internimage39) do (
  if not defined STAGE5_ENV_DIR if exist "D:\Miniconda3\envs\%%E\python.exe" (
    set "STAGE5_ENV=%%E"
    set "STAGE5_ENV_DIR=D:\Miniconda3\envs\%%E"
  )
  if not defined STAGE5_ENV_DIR if exist "E:\Anaconda\envs\%%E\python.exe" (
    set "STAGE5_ENV=%%E"
    set "STAGE5_ENV_DIR=E:\Anaconda\envs\%%E"
  )
  if not defined STAGE5_ENV_DIR if exist "%USERPROFILE%\miniconda3\envs\%%E\python.exe" (
    set "STAGE5_ENV=%%E"
    set "STAGE5_ENV_DIR=%USERPROFILE%\miniconda3\envs\%%E"
  )
  if not defined STAGE5_ENV_DIR if exist "%USERPROFILE%\anaconda3\envs\%%E\python.exe" (
    set "STAGE5_ENV=%%E"
    set "STAGE5_ENV_DIR=%USERPROFILE%\anaconda3\envs\%%E"
  )
  if not defined STAGE5_ENV_DIR if exist "%ProgramData%\miniconda3\envs\%%E\python.exe" (
    set "STAGE5_ENV=%%E"
    set "STAGE5_ENV_DIR=%ProgramData%\miniconda3\envs\%%E"
  )
  if not defined STAGE5_ENV_DIR if exist "%ProgramData%\anaconda3\envs\%%E\python.exe" (
    set "STAGE5_ENV=%%E"
    set "STAGE5_ENV_DIR=%ProgramData%\anaconda3\envs\%%E"
  )
)

if not defined STAGE5_ENV_DIR (
  echo [error] Cannot find env internimage310 or internimage39.
  echo [error] Run stage5_internimage\00_repair_mmcv_full_310.bat after creating internimage310.
  exit /b 2
)

set "CONDA_PREFIX=%STAGE5_ENV_DIR%"
set "CONDA_DEFAULT_ENV=%STAGE5_ENV%"
set "PYTHONNOUSERSITE=1"
set "PATH=%STAGE5_ENV_DIR%;%STAGE5_ENV_DIR%\Scripts;%STAGE5_ENV_DIR%\Library\bin;%STAGE5_ENV_DIR%\Library\usr\bin;%PATH%"
set "PYTHONPATH=%STAGE5_ROOT%\external_code\InternImage\detection;%PYTHONPATH%"

python -c "import sys, torch, mmcv, mmdet, pycocotools; print('[ok] Python:', sys.executable); print('[ok] torch:', torch.__version__, 'cuda=', torch.cuda.is_available()); print('[ok] mmcv:', mmcv.__version__, 'mmdet:', mmdet.__version__)"
if %ERRORLEVEL% NEQ 0 (
  echo [error] InternImage/MMDetection imports failed.
  echo [error] Run stage5_internimage\00_create_or_check_env.bat first.
  exit /b %ERRORLEVEL%
)

exit /b 0
