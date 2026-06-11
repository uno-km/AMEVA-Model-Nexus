@echo off
color 0B
echo =======================================
echo  🚀 AMEVA Model Nexus Launcher
echo =======================================

:: 1. 가상환경(venv) 존재 여부 확인 및 생성
IF NOT EXIST venv\Scripts\activate.bat (
    echo [+] Virtual environment not found. Creating 'venv'...
    python -m venv venv
    echo [+] Virtual environment created successfully!
) ELSE (
    echo [+] Virtual environment found.
)

:: 2. 가상환경 활성화
echo [+] Activating virtual environment...
call venv\Scripts\activate.bat

:: 3. 핵심 런처 파이썬 스크립트 실행
echo [+] Starting AMEVA Model Nexus...
python run_nexus.py

:: 4. 종료 후 가상환경 비활성화
echo.
echo [-] Deactivating virtual environment...
deactivate
pause
