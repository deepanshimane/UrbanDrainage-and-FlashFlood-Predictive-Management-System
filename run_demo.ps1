# run_demo.ps1 - FloodSense Demo Runner (Person C)
# Launches FastAPI backend on port 8000 and Streamlit dashboard on port 8501.

$ErrorActionPreference = "Stop"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  🌊 FLOODSENSE: AI Urban Drainage & Flood Management" -ForegroundColor Cyan
Write-Host "  (All Data Simulated for Demo and Evaluation)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = "C:\Users\deepa\AppData\Local\Programs\Python\Python313\python.exe"
if (!(Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python).Source
}
Write-Host "Using Python: $PythonExe" -ForegroundColor Green

# 1. Start FastAPI Backend in new window
Write-Host "Starting FastAPI Backend (:8000)..." -ForegroundColor Yellow
$BackendProc = Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn src.api.main:app --port 8000" -PassThru

# 2. Wait for Backend Health
$Retries = 15
$BackendReady = $false
while ($Retries -gt 0) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 2
        if ($resp.status -eq "healthy") {
            $BackendReady = $true
            break
        }
    } catch {}
    $Retries--
}

if ($BackendReady) {
    Write-Host " Backend healthy and connected on http://localhost:8000" -ForegroundColor Green
} else {
    Write-Host "⚠️ Backend warm-up taking longer than expected. Continuing to dashboard..." -ForegroundColor Yellow
}

# 3. Start Streamlit Dashboard in new window
Write-Host "Launching Streamlit Dashboard (:8501)..." -ForegroundColor Yellow
$DashboardProc = Start-Process -FilePath $PythonExe -ArgumentList "-m streamlit run dashboard/app.py --server.port 8501" -PassThru

Start-Sleep -Seconds 3
Write-Host "`nBoth services running!" -ForegroundColor Cyan
Write-Host "  • Dashboard UI:  http://localhost:8501" -ForegroundColor White
Write-Host "  • API Swagger:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "`nTo stop all demo services, press Ctrl+C or close the spawned windows." -ForegroundColor Magenta

