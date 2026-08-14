<#
.SYNOPSIS
    YOLO 학습 웹 콘솔 실행.

.DESCRIPTION
    .venv 의 Python 으로 backend/run.py 를 띄운다.
    서버는 127.0.0.1:8000 에만 바인딩된다 (localhost 전용, 인증 없음).
    멈추려면 Ctrl+C.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'

if (-not (Test-Path $VenvPython)) {
    Write-Host '가상환경(.venv)이 없다. 세팅을 먼저 해라:' -ForegroundColor Red
    Write-Host '  .\scripts\setup.ps1' -ForegroundColor Yellow
    exit 1
}

Write-Host 'YOLO 학습 웹 콘솔 시작' -ForegroundColor Cyan
Write-Host '  http://127.0.0.1:8000' -ForegroundColor Yellow
Write-Host '  멈추려면 Ctrl+C'
Write-Host ''

& $VenvPython (Join-Path $Root 'backend\run.py')
exit $LASTEXITCODE
