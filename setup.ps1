# One-command setup for Windows PowerShell.
# Usage: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "1/4 Creating virtual environment..."
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "2/4 Installing requirements..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "3/4 Downloading dataset (about 1.1 GB, first run only)..."
& .\.venv\Scripts\python.exe download_data.py

Write-Host "4/4 Checking environment..."
& .\.venv\Scripts\python.exe -m src.check_env

Write-Host "Done. Activate the environment with:  .venv\Scripts\activate"