# 인터넷이 되는 PC 에서 실행해 단독망 반입 패키지를 만든다.
#
#   .\scripts\build_bundle.ps1
#
# 만들어진 bundle\ 폴더를 통째로 대상 PC 로 옮기고 bundle\install.ps1 을 실행하면 된다.

[CmdletBinding()]
param(
    [string]$OutDir = "bundle",
    [switch]$Cpu,             # NVIDIA GPU 가 없는 대상용 (CPU 전용 torch)
    [switch]$SkipWheels,      # 휠 다운로드 건너뛰기 (구조만 확인할 때)
    [switch]$SkipPrereq,      # VC++/파이썬 설치 파일 내려받기 건너뛰기
    [switch]$SkipFrontend,    # 프론트엔드 빌드 건너뛰기
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bundle = Join-Path $root $OutDir

$TorchIndex = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu128" }
$PythonInstaller = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$VcRedist = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

function Resolve-Python {
    # 실행 파일과 인자를 분리해서 돌려준다.
    # 하나의 문자열로 두고 공백으로 쪼개면 "C:\Program Files\...\python.exe" 같은 경로가 두 토막 난다.
    $candidates = @()
    if ($PythonPath) {
        if (-not (Test-Path $PythonPath)) { throw "-PythonPath 경로가 없습니다: $PythonPath" }
        $candidates += ,@{ Exe = $PythonPath; Args = @() }
    }
    $candidates += ,@{ Exe = "py"; Args = @("-3.11") }
    $candidates += ,@{ Exe = "py"; Args = @("-3") }
    $candidates += ,@{ Exe = "python"; Args = @() }

    foreach ($c in $candidates) {
        if (-not (Test-Path $c.Exe) -and -not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $v = & $c.Exe @($c.Args) -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v -match "^3\.(9|1[0-9])$") { return $c }
        } catch { continue }
    }
    throw "Python 3 을 찾지 못했습니다. -PythonPath 로 직접 지정하세요."
}

Write-Step "준비"
$py = Resolve-Python
Write-Host "빌드에 쓸 Python: $($py.Exe) $($py.Args -join ' ')"
Write-Host "torch 인덱스: $TorchIndex"

New-Item -ItemType Directory -Force -Path $bundle | Out-Null
foreach ($sub in @("wheels", "weights", "fonts", "prereq", "frontend-dist")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $bundle $sub) | Out-Null
}

# --- 1) 휠하우스 -------------------------------------------------------------
if (-not $SkipWheels) {
    Write-Step "의존성 휠 내려받기 (수 GB, 시간이 걸립니다)"
    $wheels = Join-Path $bundle "wheels"
    $req = Join-Path $root "requirements.txt"

    # 반드시 한 번의 해석으로 받는다.
    #
    # torch 를 따로 받고 나머지를 따로 받으면, 두 번째 해석에서 ultralytics 의 torch 의존성이
    # PyPI 기준으로 다시 풀려 **CPU 빌드 torch 가 추가로 딸려 들어온다**. 실제로 그렇게 됐었다
    # (torch 2.11.0+cu128 과 torch 2.13.0 이 동시에 들어옴 → 3GB 낭비 + 잘못된 휠이 뽑힐 위험).
    # --extra-index-url 로 두 인덱스를 동시에 보게 하고 한 번에 해석해야 한다.
    #
    # --only-binary=:all: — 대상 PC 에는 컴파일러가 없으므로 소스 배포본이 섞이면 안 된다.
    # -Cpu 는 CPU 인덱스를 보게 하지만 requirements 의 핀은 +cu128 이다.
    # 그대로 두면 CPU 인덱스에서 +cu128 빌드를 못 찾아 실패한다 → 핀에서 로컬 버전 태그를 떼어낸다.
    if ($Cpu) {
        $req = Join-Path $env:TEMP "yoloconsole-requirements-cpu.txt"
        (Get-Content (Join-Path $root "requirements.txt")) -replace '\+cu\d+', '' | Set-Content $req -Encoding ASCII
        Write-Host "CPU 빌드용으로 torch 핀에서 +cuXXX 를 제거했습니다: $req"
    }

    Write-Host "requirements 를 한 번에 해석합니다 (PyPI + $TorchIndex)"
    & $py.Exe @($py.Args) -m pip download -r $req `
        --dest $wheels --only-binary=:all: --extra-index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "휠 다운로드 실패" }

    # 같은 패키지의 다른 빌드가 섞이면 대상 PC 에서 엉뚱한 게 설치될 수 있다.
    $dupes = Get-ChildItem $wheels -Filter *.whl |
        Group-Object { ($_.Name -split "-")[0].ToLower() } |
        Where-Object { $_.Count -gt 1 }
    if ($dupes) {
        Write-Host "같은 패키지의 휠이 여러 개 있습니다:" -ForegroundColor Yellow
        foreach ($d in $dupes) { Write-Host "  $($d.Name): $($d.Group.Name -join ', ')" -ForegroundColor Yellow }
        throw "휠하우스에 중복 빌드가 있습니다. wheels 폴더를 비우고 다시 실행하세요."
    }

    $size = (Get-ChildItem $wheels -File | Measure-Object Length -Sum).Sum / 1GB
    Write-Host ("휠 {0}개, {1:N2} GB" -f (Get-ChildItem $wheels -File).Count, $size)
}

# --- 2) 사전 요구 설치 파일 ---------------------------------------------------
if (-not $SkipPrereq) {
    Write-Step "사전 요구 설치 파일 내려받기"
    $prereq = Join-Path $bundle "prereq"

    # VC++ 재배포 패키지는 반드시 필요하다.
    # torch_cpu.dll 이 MSVCP140.dll / VCRUNTIME140.dll / VCRUNTIME140_1.dll 을 요구하는데
    # torch 휠에도 파이썬 설치 폴더에도 MSVCP140.dll 이 없다 → 순정 PC 에서 import torch 가 실패한다.
    foreach ($item in @(
        @{ Url = $VcRedist;        Name = "VC_redist.x64.exe" },
        @{ Url = $PythonInstaller; Name = "python-3.11.9-amd64.exe" }
    )) {
        $dest = Join-Path $prereq $item.Name
        if (Test-Path $dest) { Write-Host "이미 있음: $($item.Name)"; continue }
        Write-Host "받는 중: $($item.Name)"
        Invoke-WebRequest -Uri $item.Url -OutFile $dest -UseBasicParsing
    }
}

# --- 3) 사전학습 가중치 -------------------------------------------------------
Write-Step "사전학습 가중치 확인"
$weights = Join-Path $bundle "weights"
$pt = Get-ChildItem $weights -Filter *.pt -ErrorAction SilentlyContinue
if ($pt) {
    Write-Host "$($pt.Count) 개 있음: $($pt.Name -join ', ')"
} else {
    Write-Host "가중치가 없습니다. scripts\fetch_weights.py 를 먼저 실행하세요." -ForegroundColor Yellow
}

# --- 4) 프론트엔드 ------------------------------------------------------------
if (-not $SkipFrontend) {
    Write-Step "프론트엔드 빌드"
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($npm) {
        Push-Location (Join-Path $root "frontend")
        try {
            # $ErrorActionPreference="Stop" 은 네이티브 명령의 종료 코드를 잡지 않는다.
            # 확인하지 않으면 빌드가 깨져도 예전 dist 를 그대로 반입하고 "완료"를 찍는다.
            if (-not (Test-Path "node_modules")) {
                & npm install --no-audit --no-fund
                if ($LASTEXITCODE -ne 0) { throw "npm install 실패" }
            }
            & npm run build
            if ($LASTEXITCODE -ne 0) { throw "프론트엔드 빌드 실패 — 예전 dist 를 반입하지 않도록 중단합니다." }
        } finally { Pop-Location }
    } else {
        Write-Host "npm 이 없어 건너뜁니다. 기존 frontend\dist 를 그대로 씁니다." -ForegroundColor Yellow
    }
}
$dist = Join-Path $root "frontend\dist"
if (Test-Path $dist) {
    $target = Join-Path $bundle "frontend-dist"
    Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item $dist $target -Recurse
    Write-Host "frontend\dist 복사 완료"
} else {
    Write-Host "frontend\dist 가 없습니다. 대상 PC 에서 화면이 뜨지 않습니다." -ForegroundColor Yellow
}

# --- 5) lock 파일 -------------------------------------------------------------
Write-Step "requirements.lock.txt 생성"
Copy-Item (Join-Path $root "requirements.txt") (Join-Path $bundle "requirements.lock.txt") -Force

Write-Step "완료"
Write-Host "반입할 폴더: $bundle"
Write-Host "대상 PC 에서 bundle\install.ps1 을 관리자 권한으로 실행하세요."
Write-Host ""
Write-Host "NVIDIA 그래픽 드라이버는 pip 로 배포할 수 없습니다." -ForegroundColor Yellow
Write-Host "GPU 를 쓸 거라면 드라이버 설치 파일을 직접 bundle\prereq\ 에 넣어 함께 반입하세요." -ForegroundColor Yellow
