@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Could not create the Python environment. Install Python 3.11 first.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo Could not install project requirements. Check your internet connection.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py
if errorlevel 1 pause
