@echo off
setlocal
set "ROOT=D:\detr_Q3"
set "ENV_NAME=daq"
set "PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_EXTRA=https://pypi.mirrors.ustc.edu.cn/simple"
set "PIP_ALIYUN=https://mirrors.aliyun.com/pypi/simple"
set "TORCH_INDEX=https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"
set "CONDA_MAIN=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
set "CONDA_MSYS2=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2"
set "CONDA_PYTORCH=https://mirrors.sustech.edu.cn/anaconda-extra/cloud/pytorch"
set "CONDA_NVIDIA=https://mirrors.sustech.edu.cn/anaconda-extra/cloud/nvidia"
set "CONDA_PYTORCH_FALLBACK=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch"

rem Avoid an already-active Anaconda base redirecting another condabin\conda.bat.
set "CONDA_EXE="
set "_CONDA_EXE="
set "_CE_CONDA="
set "_CE_M="

cd /d "%ROOT%"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

call :find_conda
if not defined CONDA_CMD (
  echo [info] Anaconda/Miniconda not found. Installing Miniconda from China mirrors.
  powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\stage5_5hosts\install_miniconda.ps1"
  if errorlevel 1 exit /b 1
)

call :find_conda
if not defined CONDA_CMD (
  echo [error] Cannot find conda.exe after Miniconda setup.
  exit /b 2
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [info] conda env %ENV_NAME% is missing or not Python 3.10; recreating it from Tsinghua mirrors.
  call "%CONDA_CMD%" env remove -n %ENV_NAME% -y >nul 2>nul
  call "%CONDA_CMD%" create -n %ENV_NAME% python=3.10 pip -y --override-channels -c %CONDA_MAIN% -c %CONDA_MSYS2%
  if errorlevel 1 exit /b 1
)

call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -c "import sys; print(sys.executable); print(sys.version)"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip install --upgrade pip -i %PIP_INDEX% --extra-index-url %PIP_EXTRA% --timeout 120 --retries 10
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import torch" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [info] installing CUDA PyTorch 12.1 from SUSTech conda pytorch/nvidia mirrors.
  call "%CONDA_CMD%" install -n %ENV_NAME% pytorch torchvision pytorch-cuda=12.1 -y --override-channels -c %CONDA_PYTORCH% -c %CONDA_NVIDIA% -c %CONDA_MAIN% -c %CONDA_MSYS2%
  if errorlevel 1 (
    echo [warn] SUSTech conda PyTorch install failed; trying Tsinghua pytorch + SUSTech nvidia.
    call "%CONDA_CMD%" install -n %ENV_NAME% pytorch torchvision pytorch-cuda=12.1 -y --override-channels -c %CONDA_PYTORCH_FALLBACK% -c %CONDA_NVIDIA% -c %CONDA_MAIN% -c %CONDA_MSYS2%
  )
  if errorlevel 1 (
    echo [warn] conda PyTorch mirrors failed; trying SJTU pip PyTorch mirror.
    call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip install torch torchvision --index-url %TORCH_INDEX% --timeout 90 --retries 3 --progress-bar on
  )
  if errorlevel 1 exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [info] existing torch is not CUDA-enabled; reinstalling CUDA PyTorch 12.1 from SUSTech conda pytorch/nvidia mirrors.
  echo [info] removing pip-installed CPU torch/torchvision if present.
  call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip uninstall -y torch torchvision torchaudio
  call "%CONDA_CMD%" remove -n %ENV_NAME% pytorch torchvision pytorch-cuda -y >nul 2>nul
  call "%CONDA_CMD%" install -n %ENV_NAME% pytorch torchvision pytorch-cuda=12.1 -y --override-channels -c %CONDA_PYTORCH% -c %CONDA_NVIDIA% -c %CONDA_MAIN% -c %CONDA_MSYS2%
  if errorlevel 1 (
    echo [warn] SUSTech conda PyTorch reinstall failed; trying Tsinghua pytorch + SUSTech nvidia.
    call "%CONDA_CMD%" install -n %ENV_NAME% pytorch torchvision pytorch-cuda=12.1 -y --override-channels -c %CONDA_PYTORCH_FALLBACK% -c %CONDA_NVIDIA% -c %CONDA_MAIN% -c %CONDA_MSYS2%
  )
  if errorlevel 1 (
    echo [warn] conda PyTorch mirrors failed; trying SJTU pip PyTorch mirror.
    call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip install --upgrade --force-reinstall torch torchvision --index-url %TORCH_INDEX% --timeout 90 --retries 3 --progress-bar on
  )
  if errorlevel 1 exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import ultralytics, matplotlib, numpy, yaml" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo [info] installing Stage 5 Python packages one by one from PyPI mirrors.
  call :pip_install numpy
  if errorlevel 1 exit /b 1
  call :pip_install pyyaml
  if errorlevel 1 exit /b 1
  call :pip_install matplotlib
  if errorlevel 1 exit /b 1
  call :pip_install opencv-python
  if errorlevel 1 exit /b 1
  call :pip_install pillow
  if errorlevel 1 exit /b 1
  call :pip_install tqdm
  if errorlevel 1 exit /b 1
  call :pip_install requests
  if errorlevel 1 exit /b 1
  call :pip_install scipy
  if errorlevel 1 exit /b 1
  call :pip_install psutil
  if errorlevel 1 exit /b 1
  call :pip_install py-cpuinfo
  if errorlevel 1 exit /b 1
  call :pip_install pandas
  if errorlevel 1 exit /b 1
  call :pip_install seaborn
  if errorlevel 1 exit /b 1
  call :pip_install_no_deps polars
  if errorlevel 1 exit /b 1
  call :pip_install_no_deps ultralytics-thop
  if errorlevel 1 exit /b 1
  call :pip_install_no_deps ultralytics
  if errorlevel 1 exit /b 1
)

call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -c "import sys, torch, ultralytics, numpy; assert sys.version_info[:2] == (3, 10), sys.executable; assert torch.cuda.is_available(), 'CUDA unavailable; update NVIDIA driver or check GPU'; x=torch.randn((512,512),device='cuda'); y=(x@x).sum(); torch.cuda.synchronize(); print(sys.executable); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); print('ultralytics', ultralytics.__version__); print('numpy', numpy.__version__); print('cuda_smoke_sum', round(float(y.detach().cpu()), 4))"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [ok] Stage 5 %ENV_NAME% mirror environment is ready.
exit /b 0

:find_conda
set "CONDA_CMD="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\anaconda3\Scripts\conda.exe"
exit /b 0

:pip_install
echo [info] pip installing %*
call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip install --prefer-binary --progress-bar on --timeout 60 --retries 3 -i %PIP_INDEX% --extra-index-url %PIP_EXTRA% %*
if errorlevel 1 (
  echo [warn] Tsinghua/USTC PyPI mirrors failed for %*; trying Aliyun PyPI mirror.
  call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip install --prefer-binary --progress-bar on --timeout 60 --retries 3 -i %PIP_ALIYUN% %*
)
if errorlevel 1 exit /b 1
exit /b 0

:pip_install_no_deps
echo [info] pip installing %* without dependencies
call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip install --no-deps --prefer-binary --progress-bar on --timeout 60 --retries 3 -i %PIP_INDEX% --extra-index-url %PIP_EXTRA% %*
if errorlevel 1 (
  echo [warn] Tsinghua/USTC PyPI mirrors failed for %*; trying Aliyun PyPI mirror.
  call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME% python -m pip install --no-deps --prefer-binary --progress-bar on --timeout 60 --retries 3 -i %PIP_ALIYUN% %*
)
if errorlevel 1 exit /b 1
exit /b 0
