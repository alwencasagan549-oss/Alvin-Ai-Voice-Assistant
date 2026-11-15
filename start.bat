@echo off
REM Alvin AI -- Holographic UI Launcher.
REM
REM First-run experience: starts the FastAPI backend and keeps the PySide6
REM Holographic UI hidden so nothing pops up on launch. The HUD appears only
REM when the user says "Alvin, wake up" (the mic stays live in the background).
REM
REM Close the window or press Ctrl+C to stop the backend.

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo Checking for PySide6...
"%PY%" -c "import PySide6" 2>nul
if errorlevel 1 (
    echo Installing PySide6...
    "%PY%" -m pip install PySide6 sounddevice websockets
)

echo Starting Alvin AI... (say "Alvin, wake up" to show the UI)
"%PY%" run-all.py --hud --hidden

echo.
echo Alvin has shut down.
pause