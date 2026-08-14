# 단독망(인터넷 없음) PC 에 YOLO 학습 콘솔을 설치한다.
# 관리자 권한 PowerShell 에서 실행:
#
#   .\install.ps1
#
# 아무것도 깔려 있지 않은 순정 Windows 10/11 x64 를 전제로 한다.

[CmdletBinding()]
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\YoloConsole",
    [switch]$SkipPython,      # 이미 Python 3.11 이 있는 경우
    [switch]$SkipVcRedist     # 이미 VC++ 재배포 패키지가 있는 경우
)

$ErrorActionPreference = "Stop"
$bundle = $PSScriptRoot
$source = Split-Path -Parent $bundle   # bundle 과 나란히 있는 앱 소스

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host $msg -ForegroundColor Red }

# --- 0) 사전 점검 -------------------------------------------------------------
Write-Step "환경 점검"
$isAdmin = [Security.Principal.WindowsPrincipal]::new(
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "관리자 권한으로 실행해야 합니다. PowerShell 을 '관리자로 실행' 하세요." }
if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") { throw "x64 Windows 에서만 설치할 수 있습니다." }
Write-Host "OS: $((Get-CimInstance Win32_OperatingSystem).Caption)"
Write-Host "설치 위치: $InstallDir"

# --- 1) VC++ 재배포 패키지 ----------------------------------------------------
# 이 단계를 건너뛰면 뒤가 전부 무의미하다.
# torch_cpu.dll 이 MSVCP140.dll 을 요구하는데 torch 휠에도 파이썬 설치 폴더에도 그 DLL 이 없다.
# 순정 Windows 에는 보통 없으므로 여기서 깔지 않으면 `import torch` 가 실패한다.
if (-not $SkipVcRedist) {
    Write-Step "VC++ 2015-2022 재배포 패키지 설치"
    $vc = Join-Path $bundle "prereq\VC_redist.x64.exe"
    if (Test-Path $vc) {
        $p = Start-Process $vc -ArgumentList "/quiet", "/norestart" -Wait -PassThru
        # 3010 = 재부팅 필요. 설치 자체는 성공이다.
        if ($p.ExitCode -notin @(0, 1638, 3010)) { throw "VC++ 재배포 패키지 설치 실패 (코드 $($p.ExitCode))" }
        if ($p.ExitCode -eq 3010) { Write-Warn "설치 후 재부팅이 필요할 수 있습니다." }
        Write-Host "완료"
    } else {
        Write-Warn "prereq\VC_redist.x64.exe 가 없습니다. import torch 가 실패할 수 있습니다."
    }
}

# --- 2) Python ---------------------------------------------------------------
function Find-Python311 {
    foreach ($cand in @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
                        "C:\Program Files\Python311\python.exe")) {
        if (Test-Path $cand) { return $cand }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $p = & py -3.11 -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
        } catch {}
    }
    return $null
}

Write-Step "Python 3.11 준비"
$python = Find-Python311
if (-not $python -and -not $SkipPython) {
    $installer = Join-Path $bundle "prereq\python-3.11.9-amd64.exe"
    if (-not (Test-Path $installer)) { throw "prereq\python-3.11.9-amd64.exe 가 없습니다." }
    Write-Host "Python 3.11.9 설치 중..."
    $p = Start-Process $installer -ArgumentList `
        "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1", "Include_test=0" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Python 설치 실패 (코드 $($p.ExitCode))" }
    $python = Find-Python311
}
if (-not $python) { throw "Python 3.11 을 찾지 못했습니다." }
Write-Host "Python: $python"

# --- 3) 가상환경 + 오프라인 설치 ----------------------------------------------
# 앱 소스를 덮어쓰기 **전에** 의존성부터 깐다.
# 반대 순서로 하면, 재설치가 중간에 실패했을 때 새 코드 + 구 의존성이 섞인 상태로 남는다.
#
# 설치 위치는 Program Files 가 아니라 사용자 폴더다. 앱이 storage/, app.db(WAL) 에 쓰기 때문에
# 시스템 폴더에 설치하면 일반 사용자 실행이 권한으로 막힌다.
Write-Step "가상환경 생성 및 패키지 설치 (오프라인)"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$venv = Join-Path $InstallDir ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "가상환경 생성 실패" }
}
$vpy = Join-Path $venv "Scripts\python.exe"

$wheels = Join-Path $bundle "wheels"
$lock = Join-Path $bundle "requirements.lock.txt"
if (-not (Test-Path $wheels)) { throw "wheels 폴더가 없습니다." }
if (-not (Test-Path $lock)) { throw "requirements.lock.txt 가 없습니다." }

& $vpy -m pip install --no-index --find-links $wheels --upgrade pip 2>&1 | Out-Null
& $vpy -m pip install --no-index --find-links $wheels -r $lock
if ($LASTEXITCODE -ne 0) { throw "패키지 설치 실패. 휠이 모두 들어 있는지 확인하세요." }

Write-Host "의존성 정합성 확인 (pip check)"
& $vpy -m pip check
if ($LASTEXITCODE -ne 0) { Write-Warn "pip check 가 경고를 냈습니다. 위 메시지를 확인하세요." }

# 여기서 임포트가 깨지면 앱 소스는 아직 건드리지 않은 상태다.
Write-Host "핵심 패키지 임포트 확인"
& $vpy -c "import torch, torchvision, cv2, ultralytics, fastapi" 2>&1 | Out-String | Write-Host
if ($LASTEXITCODE -ne 0) {
    Write-Fail "임포트 실패 — VC++ 재배포 패키지가 빠졌을 가능성이 큽니다."
    Write-Fail "앱 파일은 아직 복사하지 않았으므로 기존 설치는 그대로입니다."
    exit 1
}

# --- 4) 앱 배치 --------------------------------------------------------------
Write-Step "앱 파일 복사"
foreach ($item in @("backend", "frontend", "scripts", "examples", "README.md", "requirements.txt")) {
    $src = Join-Path $source $item
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $InstallDir $item) -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "bundle\weights") | Out-Null
Copy-Item (Join-Path $bundle "weights\*") (Join-Path $InstallDir "bundle\weights") -Force -ErrorAction SilentlyContinue

# 앱이 실제로 읽는 경로는 frontend\dist 다.
$distSrc = Join-Path $bundle "frontend-dist"
if (Test-Path $distSrc) {
    $distDst = Join-Path $InstallDir "frontend\dist"
    New-Item -ItemType Directory -Force -Path (Split-Path $distDst) | Out-Null
    Remove-Item $distDst -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item $distSrc $distDst -Recurse
    Write-Host "프론트엔드 배치 완료"
} else {
    Write-Warn "frontend-dist 가 없습니다. 화면이 뜨지 않습니다."
}

# --- 5) 폰트 ------------------------------------------------------------------
$font = Join-Path $bundle "fonts\Arial.ttf"
if (Test-Path $font) {
    $cfg = Join-Path $env:APPDATA "Ultralytics"
    New-Item -ItemType Directory -Force -Path $cfg | Out-Null
    Copy-Item $font $cfg -Force
    Write-Host "폰트 배치 완료"
}

# --- 6) 실행 스크립트 ---------------------------------------------------------
Write-Step "실행 스크립트 생성"
$start = Join-Path $InstallDir "start.ps1"
@"
# YOLO 학습 콘솔 실행
`$env:YOLO_OFFLINE = "true"
`$env:YOLO_AUTOINSTALL = "false"
Start-Process "http://127.0.0.1:8000"
& "$venv\Scripts\python.exe" "$InstallDir\backend\run.py"
"@ | Set-Content -Path $start -Encoding UTF8

$shortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "YOLO 학습 콘솔.lnk"
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($shortcut)
$lnk.TargetPath = "powershell.exe"
$lnk.Arguments = "-ExecutionPolicy Bypass -File `"$start`""
$lnk.WorkingDirectory = $InstallDir
$lnk.Save()
Write-Host "바탕화면 바로가기 생성"

# --- 7) 자체 점검 -------------------------------------------------------------
# 여기서 조용히 넘어가면 나중에 "왜 이렇게 느리지" 로 돌아온다.
Write-Step "설치 점검"
$check = @'
import sys
ok = True
try:
    import torch, torchvision, cv2, ultralytics, fastapi
    print(f"  ultralytics {ultralytics.__version__}")
    print(f"  torch {torch.__version__} (CUDA 빌드: {torch.version.cuda})")
    print(f"  opencv {cv2.__version__}")
except Exception as e:
    print(f"  임포트 실패: {type(e).__name__}: {e}")
    sys.exit(1)

if torch.cuda.is_available():
    print(f"  GPU 사용 가능: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()}장)")
else:
    print("  GPU 사용 불가 -> CPU 로 동작합니다 (학습이 매우 느립니다)")
    print("  NVIDIA 그래픽 드라이버가 설치되어 있는지 확인하세요.")
sys.exit(0)
'@
$check | & $vpy -
if ($LASTEXITCODE -ne 0) {
    Write-Fail "점검 실패 - 위 오류를 확인하세요. VC++ 재배포 패키지가 빠졌을 가능성이 큽니다."
    exit 1
}

Write-Step "설치 완료"
Write-Host "바탕화면의 'YOLO 학습 콘솔' 바로가기로 실행하거나:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$start`""
Write-Host "브라우저에서 http://127.0.0.1:8000 을 엽니다."
