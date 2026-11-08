@echo off
setlocal
cd /d "%~dp0"
set "WORKSPACE=%LOCALAPPDATA%\EEGReviewer\Workspace"
set "APP=%~dp0dist\EEGReviewer\EEGReviewer.exe"
if exist "%APP%" (
    start "" "%APP%" --workspace "%WORKSPACE%"
    exit /b
)
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%~dp0launch_reviewer.py" --workspace "%WORKSPACE%"
if errorlevel 1 pause
