@echo off
setlocal enabledelayedexpansion

echo ================================================
echo 🚀 SENTRY-AI - Windows Launcher
echo ================================================
echo.

:: Check for Python
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=py
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=python
    ) else (
        echo ❌ Python not found. Please install Python.
        pause
        exit /b 1
    )
)

echo 📋 Using %PY_CMD%...
echo.

:: Check for venv
if not exist venv (
    echo 🌍 Creating virtual environment...
    %PY_CMD% -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created.
) else (
    echo ⚠ Virtual environment already exists.
)
echo.

:: Activate venv and install dependencies
echo 🔌 Activating venv and checking dependencies...
call venv\Scripts\activate
%PY_CMD% -m pip install --upgrade pip
%PY_CMD% -m pip install -r requirements.txt
echo ✓ Dependencies ready.
echo.

:: Check for missing directories
echo 📁 Ensuring directory structure...
if not exist models mkdir models
if not exist outputs mkdir outputs
if not exist data mkdir data
if not exist logs mkdir logs
echo ✓ Directories checked.
echo.

:: Final Check
if not exist models\log_transformer.keras (
    echo ⚠ No trained model found in models/log_transformer.keras.
    echo   You may need to train a model first or use the dashboard to trigger training.
)
echo.

echo ================================================
echo ✨ Setup Complete! Starting SENTRY-AI Backend...
echo ================================================
echo.
echo Dashboard: http://localhost:5000
echo API:       http://localhost:5000/api/health
echo.
echo Press Ctrl+C to stop the server.
echo.

%PY_CMD% backend.py

pause
