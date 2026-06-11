@echo off
set "STAGE5_ROOT=D:\detr_Q3"
cd /d "%STAGE5_ROOT%"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
set "CONDA_ROOT="

rem Avoid an already-active Anaconda base redirecting Python or conda commands.
set "CONDA_EXE="
set "_CONDA_EXE="
set "_CE_CONDA="
set "_CE_M="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"

if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not defined CONDA_ROOT if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=%ProgramData%\miniconda3"
if not defined CONDA_ROOT if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=%ProgramData%\anaconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\miniconda3\envs\daq310\python.exe" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\miniconda3\envs\daq\python.exe" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\anaconda3\envs\daq310\python.exe" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\anaconda3\envs\daq\python.exe" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not defined CONDA_ROOT if exist "%ProgramData%\miniconda3\envs\daq310\python.exe" set "CONDA_ROOT=%ProgramData%\miniconda3"
if not defined CONDA_ROOT if exist "%ProgramData%\miniconda3\envs\daq\python.exe" set "CONDA_ROOT=%ProgramData%\miniconda3"
if not defined CONDA_ROOT if exist "%ProgramData%\anaconda3\envs\daq310\python.exe" set "CONDA_ROOT=%ProgramData%\anaconda3"
if not defined CONDA_ROOT if exist "%ProgramData%\anaconda3\envs\daq\python.exe" set "CONDA_ROOT=%ProgramData%\anaconda3"

if not defined CONDA_ROOT (
  echo [error] Cannot find Anaconda/Miniconda conda.exe.
  echo [error] Run stage5_5hosts\00_create_or_check_env.bat first.
  exit /b 2
)

set "STAGE5_ENV="
set "STAGE5_ENV_DIR="

if exist "%CONDA_ROOT%\envs\daq310\python.exe" (
  set "STAGE5_ENV=daq310"
  set "STAGE5_ENV_DIR=%CONDA_ROOT%\envs\daq310"
)

if not defined STAGE5_ENV_DIR if exist "%CONDA_ROOT%\envs\daq\python.exe" (
  set "STAGE5_ENV=daq"
  set "STAGE5_ENV_DIR=%CONDA_ROOT%\envs\daq"
)

if not defined STAGE5_ENV (
  echo [error] Cannot find conda env daq310 or daq under %CONDA_ROOT%\envs.
  echo [error] Run stage5_5hosts\00_create_or_check_env.bat first.
  exit /b 1
)

set "CONDA_PREFIX=%STAGE5_ENV_DIR%"
set "CONDA_DEFAULT_ENV=%STAGE5_ENV%"
set "CONDA_PROMPT_MODIFIER=(%STAGE5_ENV%) "
set "PATH=%STAGE5_ENV_DIR%;%STAGE5_ENV_DIR%\Scripts;%STAGE5_ENV_DIR%\Library\bin;%STAGE5_ENV_DIR%\Library\usr\bin;%CONDA_ROOT%\condabin;%PATH%"

python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.executable" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [error] conda env %STAGE5_ENV% did not expose Python 3.10.
  echo [error] Run stage5_5hosts\00_create_or_check_env.bat to repair the env.
  where python
  python -c "import sys; print(sys.executable); print(sys.version)"
  exit /b 3
)

echo [ok] Activated conda env %STAGE5_ENV% at %STAGE5_ENV_DIR%.
exit /b 0
