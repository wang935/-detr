@echo off
REM ============================================================
REM  DAQ-DETR  Tier-A pilot  --  RT-DETR query-distribution probe
REM  Run from an activated Python/conda env if you use one.
REM ============================================================
cd /d D:\detr_Q3

echo [1/2] Ensuring ultralytics is installed...
pip install -q ultralytics

echo.
echo [2/2] Running query-signal probe on pilot_images ...
python scripts\pilot_query_signal.py --weights rtdetr-l.pt --images pilot_images --out pilot_outputs\query_probe

echo.
echo ============================================================
echo  Done. Outputs:  pilot_outputs\query_probe\
echo    - query_features.csv       (per-image distribution features)
echo    - query_separability.json  (fire/smoke vs distractor gaps)
echo    - query_vectors.npz        (raw per-query vectors)
echo    - query_hist.png           (distribution plot, if matplotlib present)
echo  Send me query_separability.json + query_hist.png to interpret.
echo ============================================================
pause
