@echo off
REM Start both backend and frontend with a single command.
REM Usage: start-all.bat
REM
REM Bug 12/14: this wrapper now just launches the Python launcher
REM (run-all.py), which handles HOST parsing, port checks, health
REM probes, and clean process-tree teardown for all platforms. Keeping
REM ONE implementation of that logic avoids drift between .sh/.bat/.ps1.

REM Resolve the script directory (handles "Run as administrator" / network paths).
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Locate a Python interpreter, preferring the project venv.
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

REM Hand off to the unified launcher. -B keeps stale .pyc from masking edits.
exec "%PY" -B run-all.py
