@echo off
echo Starting AI Governance Tool...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    python setup.py
)

REM Start the application
echo.
echo Starting Streamlit application...
echo The application will open in your web browser.
echo.
echo Note: If you encounter file watcher errors, they can be safely ignored.
echo.
streamlit run main.py --server.runOnSave false --server.fileWatcherType none

pause
