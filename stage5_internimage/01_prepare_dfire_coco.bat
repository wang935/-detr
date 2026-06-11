@echo off
setlocal
cd /d D:\detr_Q3
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -m py_compile scripts\stage5_internimage_prepare_dfire.py scripts\stage5_internimage_runner.py scripts\stage5_internimage_export_pkl.py scripts\stage5_internimage_eval.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python scripts\stage5_internimage_prepare_dfire.py --out-dir data\stage5_internimage\dfire\annotations
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [ok] D-Fire COCO annotations prepared for InternImage.
endlocal
