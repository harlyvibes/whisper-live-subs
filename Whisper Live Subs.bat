@echo off
rem Starts Whisper Live Subs in the system tray (no console window).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: setting up... this downloads ~600 MB of libraries.
    call setup.bat || exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0WhisperLiveSubs.pyw"
