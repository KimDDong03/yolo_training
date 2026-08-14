<#
.SYNOPSIS
    YOLO 학습 웹 콘솔 세팅. clone 직후 한 번만 돌리면 된다.

.EXAMPLE
    .\scripts\setup.ps1
    NVIDIA GPU 가 있는 PC. CUDA 12.8 빌드 torch 를 깐다.

.EXAMPLE
    .\scripts\setup.ps1 -Cpu
    GPU 가 없는 PC. 학습은 되지만 매우 느리다.

.EXAMPLE
    .\scripts\setup.ps1 -SkipFrontend
    프론트엔드를 다시 빌드하지 않는다. 저장소에 커밋된 frontend/dist 를 그대로 쓴다.
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$SkipFrontend
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-LastExit([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host "실패: $What (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

Write-Host "설치 위치: $Root"

# --- 1. Python 확인 -----------------------------------------------------------
Write-Step 'Python 확인'

$PythonCmd = $null
$PythonArgs = @()

foreach ($candidate in @(@('py', @('-3.11')), @('py', @('-3')), @('python', @()))) {
    $exe = $candidate[0]
    $exeArgs = $candidate[1]
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }

    $version = & $exe @exeArgs --version 2>&1
    if ($LASTEXITCODE -ne 0) { continue }

    Write-Host "  $exe $exeArgs -> $version"
    $PythonCmd = $exe
    $PythonArgs = $exeArgs
    break
}

if (-not $PythonCmd) {
    Write-Host ''
    Write-Host 'Python 을 찾지 못했다. Python 3.11 을 설치해라:' -ForegroundColor Red
    Write-Host '  https://www.python.org/downloads/release/python-3119/'
    Write-Host '  설치할 때 "Add python.exe to PATH" 를 반드시 체크한다.'
    exit 1
}

# --- 2. 가상환경 --------------------------------------------------------------
Write-Step '가상환경(.venv) 준비'

if (Test-Path $VenvPython) {
    Write-Host '  이미 있음 — 건너뛴다'
} else {
    & $PythonCmd @PythonArgs -m venv (Join-Path $Root '.venv')
    Assert-LastExit '가상환경 생성'
    Write-Host '  생성 완료'
}

& $VenvPython -m pip install --upgrade pip --quiet
Assert-LastExit 'pip 업그레이드'

# --- 3. torch ----------------------------------------------------------------
if ($Cpu) {
    Write-Step 'torch 설치 (CPU 빌드)'
    & $VenvPython -m pip install torch torchvision
} else {
    Write-Step 'torch 설치 (CUDA 12.8 빌드) — 수 GB 라 몇 분 걸린다'
    & $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
}
Assert-LastExit 'torch 설치'

# --- 4. 나머지 의존성 ---------------------------------------------------------
Write-Step '백엔드 의존성 설치'
& $VenvPython -m pip install -r (Join-Path $Root 'requirements.txt')
Assert-LastExit 'requirements.txt 설치'

# --- 5. 사전학습 가중치 -------------------------------------------------------
Write-Step '사전학습 가중치 내려받기'
& $VenvPython (Join-Path $Root 'scripts\fetch_weights.py')
Assert-LastExit '가중치 다운로드'

# --- 6. 절대경로 이전 ---------------------------------------------------------
Write-Step '기존 데이터셋·학습 결과의 경로를 이 PC 에 맞게 고치기'
& $VenvPython (Join-Path $Root 'scripts\relocate.py')
Assert-LastExit '경로 이전'

# --- 7. 프론트엔드 ------------------------------------------------------------
Write-Step '프론트엔드'

if ($SkipFrontend) {
    Write-Host '  -SkipFrontend — 커밋된 frontend/dist 를 그대로 쓴다'
} elseif (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host '  Node.js 가 없다 — 커밋된 frontend/dist 를 그대로 쓴다 (문제 없음).'
    Write-Host '  프론트엔드 소스를 고칠 계획이라면 Node.js 20+ 를 설치해라: https://nodejs.org/'
} else {
    Push-Location (Join-Path $Root 'frontend')
    try {
        Write-Host '  npm ci'
        npm ci
        Assert-LastExit 'npm ci'
        Write-Host '  npm run build'
        npm run build
        Assert-LastExit 'npm run build'
    } finally {
        Pop-Location
    }
}

# --- 끝 -----------------------------------------------------------------------
Write-Host ''
Write-Host '세팅 완료.' -ForegroundColor Green
Write-Host ''
Write-Host '실행:'
Write-Host '  .\scripts\start.ps1' -ForegroundColor Yellow
Write-Host ''
Write-Host '그리고 브라우저에서 http://127.0.0.1:8000 을 연다.'
