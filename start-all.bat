@echo off
REM Start both backend and frontend with a single command
REM Usage: start-all.bat

REM Refuse occupied ports; do not terminate unrelated processes.
for %%P in (8000 5173) do (
    netstat -ano | findstr "LISTENING" | findstr /C:":%%P " >nul
    if not errorlevel 1 (
        echo Port %%P is already in use. Close its owner and try again.
        exit /b 1
    )
)

echo === Starting Alvin AI Dashboard ===

REM Start backend
echo Starting backend...
start /B cmd /c .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --ws auto --log-level info

REM Wait for backend
echo Waiting for backend...
:waitbackend
curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Backend ready ^^(http://localhost:8000^^)
    goto :startfrontend
)
timeout /t 2 /nobreak >nul
goto :waitbackend

:startfrontend
REM Start frontend
echo Starting frontend...
cd alvin-ai-dashboard
if exist "%ProgramFiles%\nodejs\npm.cmd" (
    start /B cmd /c ""%ProgramFiles%\nodejs\npm.cmd" run dev -- --port 5173"
) else (
    start /B cmd /c npm run dev -- --port 5173
)
cd ..

REM Wait for frontend
echo Waiting for frontend...
:waitfrontend
curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Frontend ready ^^(http://localhost:5173^^)
    goto :running
)
timeout /t 2 /nobreak >nul
goto :waitfrontend

:running
echo.
echo === ALL SERVERS RUNNING ===
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   Press Ctrl+C to stop.
echo.
pause
