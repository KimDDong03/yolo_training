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

.EXAMPLE
    .\scripts\setup.ps1 -PythonPath "C:\Python311\python.exe"
    Python 을 자동으로 못 찾을 때 직접 지정한다.
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$SkipFrontend,
    [string]$PythonPath
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

# 후보를 --version 이 아니라 실제 코드 실행으로 검증한다.
# Microsoft Store 스텁(WindowsApps\python.exe)은 --version 에도 반응하는 척하다가
# 실제로는 스토어 창만 띄우므로, sys.executable 을 찍게 해서 걸러낸다.
function Test-PythonCandidate {
    param([string]$Exe, [string[]]$PreArgs = @())

    if (-not $Exe) { return $null }
    try {
        $out = & $Exe @PreArgs -c "import sys; print('PYOK', sys.version_info.major, sys.version_info.minor, sys.executable)" 2>$null
    } catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0) { return $null }

    $line = $out | Where-Object { $_ -like 'PYOK *' } | Select-Object -First 1
    if (-not $line) { return $null }

    $parts = $line -split ' ', 4
    return [pscustomobject]@{
        Exe     = $Exe
        PreArgs = $PreArgs
        Major   = [int]$parts[1]
        Minor   = [int]$parts[2]
        Path    = $parts[3]
        Label   = "$Exe $($PreArgs -join ' ')".Trim()
    }
}

$tried = New-Object System.Collections.Generic.List[string]
$found = New-Object System.Collections.Generic.List[object]

function Add-Candidate {
    param([string]$Exe, [string[]]$PreArgs = @())
    $label = "$Exe $($PreArgs -join ' ')".Trim()
    if ($tried.Contains($label)) { return }
    $tried.Add($label) | Out-Null
    $result = Test-PythonCandidate -Exe $Exe -PreArgs $PreArgs
    if ($result) { $found.Add($result) | Out-Null }
}

if ($PythonPath) {
    Add-Candidate -Exe $PythonPath
    if ($found.Count -eq 0) {
        Write-Host ''
        Write-Host "-PythonPath 로 지정한 경로가 동작하지 않는다: $PythonPath" -ForegroundColor Red
        exit 1
    }
} else {
    # 1) py 런처 (python.org 설치본이면 보통 있다)
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @('-3.11', '-3.12', '-3.13', '-3.10', '-3')) {
            Add-Candidate -Exe 'py' -PreArgs @($v)
        }
    }

    # 2) PATH 위의 python (Store 스텁은 위 함수가 걸러낸다)
    foreach ($name in @('python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue |
               Where-Object { $_.Source -and $_.Source -notmatch 'WindowsApps' } |
               Select-Object -First 1
        if ($cmd) { Add-Candidate -Exe $cmd.Source }
    }

    # 3) 레지스트리에 등록된 설치본 (PATH 에 안 넣고 설치한 경우)
    foreach ($hive in @('HKLM:\SOFTWARE\Python\PythonCore', 'HKCU:\SOFTWARE\Python\PythonCore')) {
        if (-not (Test-Path $hive)) { continue }
        foreach ($key in Get-ChildItem $hive -ErrorAction SilentlyContinue) {
            $installPath = (Get-ItemProperty "$($key.PSPath)\InstallPath" -ErrorAction SilentlyContinue).'(default)'
            if ($installPath) { Add-Candidate -Exe (Join-Path $installPath 'python.exe') }
        }
    }

    # 4) 흔한 설치 위치 직접 탐색
    $roots = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python'),
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)},
        'C:\'
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($root in $roots) {
        foreach ($dir in Get-ChildItem $root -Directory -Filter 'Python3*' -ErrorAction SilentlyContinue) {
            Add-Candidate -Exe (Join-Path $dir.FullName 'python.exe')
        }
    }
}

# 3.10 ~ 3.13 을 우선하고, 그 안에서는 3.11 을 가장 선호한다.
$usable = $found | Where-Object { $_.Major -eq 3 -and $_.Minor -ge 9 }

if (-not $usable) {
    Write-Host ''
    Write-Host 'Python 3 을 찾지 못했다.' -ForegroundColor Red
    Write-Host ''
    Write-Host '시도한 것:'
    foreach ($t in $tried) { Write-Host "  - $t" }
    Write-Host ''
    Write-Host '해결 방법 두 가지:'
    Write-Host '  1) 이미 설치돼 있다면 경로를 직접 지정한다:' -ForegroundColor Yellow
    Write-Host '     .\scripts\setup.ps1 -PythonPath "C:\경로\python.exe"'
    Write-Host '     설치 위치를 모르면 PowerShell 에서:'
    Write-Host '       Get-ChildItem C:\ -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Select-Object -First 5 FullName'
    Write-Host ''
    Write-Host '  2) 새로 설치한다 (설치할 때 "Add python.exe to PATH" 체크):' -ForegroundColor Yellow
    Write-Host '     https://www.python.org/downloads/release/python-3119/'
    exit 1
}

$best = $usable | Sort-Object @{ Expression = { if ($_.Minor -eq 11) { 0 } elseif ($_.Minor -in 10, 12, 13) { 1 } else { 2 } } } | Select-Object -First 1

$PythonCmd = $best.Exe
$PythonArgs = $best.PreArgs
Write-Host "  찾음: Python $($best.Major).$($best.Minor) — $($best.Path)"
if ($best.Minor -lt 10 -or $best.Minor -gt 13) {
    Write-Host "  경고: 3.10~3.13 에서 검증했다. 이 버전은 torch 휠이 없을 수 있다." -ForegroundColor Yellow
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
