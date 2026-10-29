# Start both backend and frontend with a single command
# Usage: .\start-all.ps1

param(
    [switch]$Stop = $false
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir
$ProcessFile = Join-Path $ScriptDir ".alvin-servers.json"

function Write-Section { param([string]$msg) Write-Host $msg -ForegroundColor Cyan }
function Write-Ok { param([string]$msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host $msg -ForegroundColor Yellow }
function Write-Err { param([string]$msg) Write-Host "  [ERROR] $msg" -ForegroundColor Red }

function Test-PortAvailable {
    param([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return -not [bool]$conns
}

function Stop-ProcessTree {
    param([System.Diagnostics.Process]$Process)
    if ($Process -and -not $Process.HasExited) {
        & taskkill.exe /PID $Process.Id /T /F | Out-Null
    }
}

function Stop-TrackedProcess {
    param($Entry)
    $process = Get-Process -Id ([int]$Entry.Id) -ErrorAction SilentlyContinue
    if (-not $process) { return }
    if ($process.StartTime.ToUniversalTime().Ticks -ne [long]$Entry.StartTimeTicks) {
        Write-Warn "Skipping PID $($Entry.Id): it no longer belongs to this Alvin launcher."
        return
    }
    & taskkill.exe /PID ([int]$Entry.Id) /T /F | Out-Null
}

if ($Stop) {
    if (Test-Path -LiteralPath $ProcessFile) {
        $tracked = Get-Content -LiteralPath $ProcessFile -Raw | ConvertFrom-Json
        foreach ($entry in $tracked) { Stop-TrackedProcess $entry }
        Remove-Item -LiteralPath $ProcessFile -ErrorAction SilentlyContinue
        Write-Ok "Tracked Alvin servers stopped."
    } else {
        Write-Warn "No tracked Alvin servers found. Run this in the launcher window to stop it with Ctrl+C."
    }
    exit 0
}

Write-Section "=== Starting Alvin AI Dashboard ==="

foreach ($port in @(8000, 5173)) {
    if (-not (Test-PortAvailable $port)) {
        Write-Err "Port $port is already in use. Close its owner and try again."
        exit 1
    }
}

# Resolve npm.cmd path
$npmCmd = $null
$candidates = @(
    "$env:ProgramFiles\nodejs\npm.cmd",
    "${env:ProgramFiles(x86)}\nodejs\npm.cmd",
    "C:\Program Files\nodejs\npm.cmd",
    "C:\Program Files (x86)\nodejs\npm.cmd"
)
foreach ($c in $candidates) {
    if (Test-Path -LiteralPath $c) {
        $npmCmd = $c
        break
    }
}
if (-not $npmCmd) {
    $npmCmd = "npm"
}

# Start backend
Write-Warn "Starting backend..."
$backend = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--ws", "auto", "--log-level", "info" -NoNewWindow -PassThru

# Wait for backend
Write-Warn "Waiting for backend..."
$maxAttempts = 30
$attempt = 0
$backendReady = $false

while ($attempt -lt $maxAttempts) {
    $attempt++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    } catch {
        # Backend not ready yet
    }
    Start-Sleep -Seconds 2
}

if (-not $backendReady) {
    Write-Err "Backend failed to start within 60 seconds"
    Stop-ProcessTree $backend
    exit 1
}

Write-Ok "Backend ready (http://localhost:8000)"

# Start frontend
Write-Warn "Starting frontend..."
Set-Location "alvin-ai-dashboard"
$frontend = Start-Process -FilePath $npmCmd -ArgumentList "run", "dev", "--", "--port", "5173" -NoNewWindow -PassThru
Set-Location $ScriptDir

# Wait for frontend
Write-Warn "Waiting for frontend..."
$frontendReady = $false
$attempt = 0

while ($attempt -lt 15) {
    $attempt++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    } catch {
        # Frontend not ready yet
    }
    Start-Sleep -Seconds 2
}

if (-not $frontendReady) {
    Write-Err "Frontend failed to start"
    Stop-ProcessTree $frontend
    Stop-ProcessTree $backend
    exit 1
}

@(
    [PSCustomObject]@{ Id = $backend.Id; StartTimeTicks = $backend.StartTime.ToUniversalTime().Ticks },
    [PSCustomObject]@{ Id = $frontend.Id; StartTimeTicks = $frontend.StartTime.ToUniversalTime().Ticks }
) | ConvertTo-Json | Set-Content -LiteralPath $ProcessFile

Write-Ok "Frontend ready (http://localhost:5173)"

Write-Section "=== ALL SERVERS RUNNING ==="
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop, or run: .\start-all.ps1 -Stop" -ForegroundColor Yellow

# Keep running until interrupted
try {
    Wait-Process -Id $frontend.Id -ErrorAction Stop
} catch {
    Write-Err "Frontend process exited."
} finally {
    Write-Section "Shutting down..."
    Stop-ProcessTree $backend
    Stop-ProcessTree $frontend
    Remove-Item -LiteralPath $ProcessFile -ErrorAction SilentlyContinue
}
