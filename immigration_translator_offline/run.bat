@echo off
REM Windows batch script to run Immigration Translator Offline
REM Make sure Python 3.10+ is installed and in PATH

echo ========================================
echo Immigration Translator Offline
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist "venv\Lib\site-packages\fastapi" (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

REM Check if models are downloaded
if not exist "models" (
    echo Models not found. Downloading models...
    python scripts\download_models.py
    if errorlevel 1 (
        echo WARNING: Model download failed. You may need to download models manually.
    )
)

REM Set offline environment variables
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

REM Start the application
echo.
echo Starting Immigration Translator Offline...
echo Open your browser and go to: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause

