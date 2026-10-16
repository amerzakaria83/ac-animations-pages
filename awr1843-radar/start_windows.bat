@echo off
rem Double-click to set up (first run only) and launch AWR1843 Radar Studio.
cd /d "%~dp0"
if not exist .venv (
    echo Creating Python environment, this happens only once...
    py -3 -m venv .venv || python -m venv .venv || (echo Python 3.9+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
)
.venv\Scripts\python -m awr1843 gui %*
if errorlevel 1 pause
