@echo off
REM PyLCSS GUI Launcher
REM Automatically sets up virtual environment and installs dependencies if needed

REM Set the base directory (modify this path if PyLCSS is installed elsewhere)
set "PYLCSS_DIR=%~dp0"
if "%PYLCSS_DIR:~-1%"=="\" set "PYLCSS_DIR=%PYLCSS_DIR:~0,-1%"

echo PyLCSS Directory: %PYLCSS_DIR%
echo.

REM Check if Python is installed
python --version >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Attempting to install Python via winget...
    winget install --id Python.Python.3 --source winget --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo Failed to install Python automatically.
        echo Please install Python manually from https://www.python.org/downloads/
        echo Make sure to add Python to your PATH during installation.
        pause
        exit /b 1
    )
    echo Python installed successfully.
    echo Please restart this script or your command prompt to refresh the PATH.
    pause
    exit /b 0
)

REM Check if virtual environment exists
if not exist "%PYLCSS_DIR%\.venv" (
    echo Virtual environment not found. Creating one...
    python -m venv "%PYLCSS_DIR%\.venv"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Please ensure Python is installed and available in PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call "%PYLCSS_DIR%\.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Check if requirements are installed (by checking if PySide6 is available)
python -c "import PySide6" 2>nul
if errorlevel 1 (
    echo Upgrading pip...
    python -m pip install --upgrade pip

    echo Installing requirements...
    pip install -r "%PYLCSS_DIR%\requirements.txt"
    if errorlevel 1 (
        echo ERROR: Failed to install requirements.
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    )
    echo Requirements installed successfully.
    echo.
) else (
    echo Requirements already installed.
    echo.
)

REM Run the PyLCSS GUI
echo Starting PyLCSS GUI...
set "PYTHONPATH=%PYLCSS_DIR%;%PYTHONPATH%"
python "%PYLCSS_DIR%\scripts\main.py"
if errorlevel 1 (
    echo.
    echo ERROR: The application crashed or closed unexpectedly.
    pause
)

REM Deactivate virtual environment
call "%PYLCSS_DIR%\.venv\Scripts\deactivate.bat"