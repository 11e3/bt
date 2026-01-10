#!/usr/bin/env pwsh
# Code quality check script
# Run: .\check.ps1

Write-Host "🔍 Running code quality checks..." -ForegroundColor Cyan

Write-Host "`n📝 Formatting code with ruff..." -ForegroundColor Yellow
ruff format .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Formatting failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n🔧 Linting and fixing with ruff..." -ForegroundColor Yellow
ruff check . --fix --unsafe-fixes
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Linting failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n🔍 Type checking with mypy..." -ForegroundColor Yellow
# Clean mypy cache to avoid deserialization errors
Remove-Item -Path .mypy_cache -Recurse -Force -ErrorAction SilentlyContinue
$global:LASTEXITCODE = 0  # Reset exit code after cleanup
python -m mypy src/bt --strict
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Type checking failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n🧪 Running tests with coverage..." -ForegroundColor Yellow
# Ensure package is installed in editable mode
$env:PYTHONPATH = "$PWD/src"
pytest --cov=src/bt --cov-report=term-missing
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Tests failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ All checks passed!" -ForegroundColor Green
