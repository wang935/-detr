@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
call stage5_5hosts\_activate_env.bat
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
python scripts\stage5_external_negatives.py ^
  --result-root formal_results\stage5 ^
  --external-root external_data\BoWFireDataset\dataset\img ^
  --external-source "BoWFire non-fire fire-like negatives" ^
  --neutral-tokens "not_fire" ^
  --out-dir formal_results\stage5_external\bowfire ^
  --min-external-images 100
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
endlocal
