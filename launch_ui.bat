@echo off
start /B python ui.py
timeout /t 25 /nobreak > nul
start "" "http://127.0.0.1:7860"
pause