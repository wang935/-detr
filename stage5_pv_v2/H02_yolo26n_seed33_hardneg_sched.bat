@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

set "STAGE5_ENV_DIR=%USERPROFILE%\miniconda3\envs\daq310"
if not exist "%STAGE5_ENV_DIR%\python.exe" if exist "C:\Users\Administrator\miniconda3\envs\daq310\python.exe" set "STAGE5_ENV_DIR=C:\Users\Administrator\miniconda3\envs\daq310"
if not exist "%STAGE5_ENV_DIR%\python.exe" (
  echo [error] Missing %STAGE5_ENV_DIR%\python.exe
  echo [error] Expected H02 env at C:\Users\Administrator\miniconda3\envs\daq310.
  exit /b 2
)

set "CONDA_PREFIX=%STAGE5_ENV_DIR%"
set "CONDA_DEFAULT_ENV=daq310"
set "PYTHONNOUSERSITE=1"
set "PATH=%STAGE5_ENV_DIR%;%STAGE5_ENV_DIR%\Scripts;%STAGE5_ENV_DIR%\Library\bin;%STAGE5_ENV_DIR%\Library\usr\bin;%PATH%"

python -c "import sys, torch; assert sys.version_info[:2] == (3, 10), sys.executable; assert torch.cuda.is_available(); print('[ok] Activated daq310:', sys.executable); print('[ok] GPU:', torch.cuda.get_device_name(0))"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -m py_compile scripts\stage5_pv_v2_runner.py scripts\stage5_pv_v2_aggregate.py scripts\stage5_pv_v2_map_metrics.py scripts\stage5_pv_v2_compare_sched_vs_v1.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python scripts\stage5_pv_v2_runner.py --dataset dfire --family yolo26n --seed 33 --mode all --arm hardneg_sched --epochs 300 --batch 16 --workers 4 --device 0 --val false
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [ok] H02 YOLO26n seed33 hardneg_sched formal run completed.
endlocal
