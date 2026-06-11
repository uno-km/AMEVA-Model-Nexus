@echo off
color 0B
echo =======================================
echo  AMEVA Model Nexus Launcher
echo =======================================

REM 1. Check and create venv
IF NOT EXIST venv\Scripts\activate.bat (
    echo [+] Virtual environment not found. Creating venv...
    python -m venv venv
    echo [+] Virtual environment created successfully!
) ELSE (
    echo [+] Virtual environment found.
)

REM 2. Activate venv
echo [+] Activating virtual environment...
call venv\Scripts\activate.bat

REM 3. Run Python Launcher
echo [+] Starting AMEVA Model Nexus...
python run_nexus.py

REM 4. Deactivate
echo.
echo [-] Deactivating virtual environment...
deactivate
pause
