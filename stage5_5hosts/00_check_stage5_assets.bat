@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_5hosts\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('[ok] gpu=', torch.cuda.get_device_name(0))"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python -m py_compile scripts\stage5_formal_runner.py scripts\stage5_aggregate_repeats.py scripts\stage2_daq_strong.py scripts\stage3_budget_ablation.py scripts\audit_evidence_package.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_formal_runner.py --mode smoke --family yolo --seed 11 --device 0
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_formal_runner.py --mode smoke --family rtdetr --seed 11 --device 0
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] Stage 5 assets and GPU smoke checks passed.
endlocal
