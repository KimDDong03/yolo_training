# YOLO 학습 웹 콘솔

브라우저에서 데이터셋을 등록하고, 하이퍼파라미터를 폼으로 조정하고,
학습 진행·지표·로그·에폭별 예측 이미지를 실시간으로 보는 콘솔.

인터넷이 없는 단독망 운영을 전제로 만들었다. 실행 중에는 외부 요청을 하지 않는다.

---

## 요구사항

| | |
|---|---|
| OS | Windows 10 / 11 (스크립트가 PowerShell 기준) |
| Python | **3.11** 권장. [3.11.9 내려받기](https://www.python.org/downloads/release/python-3119/) — 설치할 때 **"Add python.exe to PATH"** 체크 |
| Node.js | **20+ 필요.** 화면(`frontend/dist`)은 빌드 산출물이라 저장소에 없다. `setup.ps1` 이 빌드한다 |
| GPU | NVIDIA GPU + 최신 드라이버. 없어도 `-Cpu` 로 돌아가지만 학습이 매우 느리다 |
| 디스크 | 약 6 GB (대부분 torch) |

## 빠른 시작 (3단계)

PowerShell 을 열고:

```powershell
git clone https://github.com/KimDDong03/yolo_training.git
```

```powershell
cd yolo_training
.\scripts\setup.ps1
```

```powershell
.\scripts\start.ps1
```

브라우저에서 **http://127.0.0.1:8000** 을 연다. 끝.

`setup.ps1` 이 하는 일 — 가상환경 생성 → torch(CUDA 12.8) 설치 → 나머지 의존성 설치 →
사전학습 가중치 다운로드 → 프론트엔드 빌드.
torch 가 커서 처음 한 번은 몇 분 걸린다.

처음 실행하면 데이터셋도 학습 기록도 비어 있다. 아래 순서대로 하나 만들어 보면 된다.

### setup.ps1 옵션

```powershell
.\scripts\setup.ps1 -Cpu            # NVIDIA GPU 가 없는 PC
.\scripts\setup.ps1 -SkipFrontend   # 프론트엔드를 다시 빌드하지 않는다
```

### 스크립트 실행이 막힐 때

PowerShell 실행 정책 때문에 `.ps1` 이 차단되면, 그 창에서만 한 번 풀어준다:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 처음 5분

**1. 데이터셋을 등록한다.**
왼쪽 **데이터셋 등록**에 zip 을 끌어다 놓거나, 이미 디스크에 있는 폴더면 경로를 입력한다.
`images/` + `labels/` 구조면 되고, `train`/`val` 이 없으면 알아서 8:2 로 나눈다.
등록되면 검수 리포트가 뜬다 — 라벨 없는 이미지, 좌표 이탈, 박스 크기 분포 같은 것들.

**2. 학습을 돌린다.**
**새 학습**에서 모델 경로(기본값이 `bundle/weights/yolo11n.pt`)와 에폭·이미지 크기·배치를 정하고 시작한다.
작은 데이터셋이면 에폭 3 / 이미지 크기 160 / 배치 8 로 1분 안에 한 바퀴 돌려볼 수 있다.

**3. 도는 걸 본다.**
오른쪽에 mAP·손실 차트와 로그가 실시간으로 흐르고, 왼쪽 **예측** 탭에서 에폭 슬라이더로
박스가 정확해지는 과정을 훑을 수 있다. 끝나면 **플롯** 탭에 혼동 행렬·PR 곡선이 채워지고,
**추론** 탭에서 임의 이미지로 시험해 볼 수 있다.

---

## 저장소에 들어 있는 것 / 없는 것

소스만 커밋한다. 나머지는 전부 스크립트가 만들거나 앱이 실행하면서 만든다.

| | 포함 | 비고 |
|---|---|---|
| 백엔드·프론트엔드 소스 | O | |
| `scripts/`, `bundle/install.ps1` | O | 세팅·반입 스크립트 |
| `frontend/dist` | X | 빌드 산출물. `setup.ps1` 이 `npm run build` 로 만든다 |
| `storage/`, `app.db` | X | 데이터셋·학습 결과·DB. 앱이 실행하면서 만든다 |
| `bundle/weights/*.pt` | X | `scripts/fetch_weights.py` 가 받는다 (setup 이 자동 실행) |
| `bundle/wheels/`, `bundle/prereq/` | X | 약 3 GB. `scripts/build_bundle.ps1` 이 받는다 |
| `node_modules`, `.venv` | X | setup 이 만든다 |

프론트엔드 소스를 고쳤으면 `npm run build` 만 다시 하면 된다. 커밋할 것은 없다.

## 폴더를 옮겼다면

`storage/` 안의 데이터셋 메타와 실행 설정에는 만들어질 당시의 절대경로가 들어 있다.
저장소 폴더를 통째로 옮겼다면 이걸 고쳐야 한다.

```powershell
.\.venv\Scripts\python.exe scripts\relocate.py
```

같은 경로에서 여러 번 돌려도 안전하다.

## 문제 해결

**"Python 3 을 찾지 못했다" — 분명히 Python 은 깔려 있는데**

Python 이 PATH 에 없거나, PATH 의 `python` 이 Microsoft Store 로 넘기는 껍데기라서 그렇다.
설치 경로를 직접 지정하면 된다.

```powershell
.\scripts\setup.ps1 -PythonPath "C:\Users\<사용자>\AppData\Local\Programs\Python\Python311\python.exe"
```

설치 위치를 모르면 찾아본다:

```powershell
Get-ChildItem C:\ -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Select-Object -First 5 FullName
```

`setup.ps1` 은 실패할 때 시도한 후보를 전부 출력하므로 그 목록도 참고가 된다.
(자동 탐색은 `py` 런처 → PATH → 레지스트리 → 흔한 설치 폴더 순으로 본다.)

**모델 자동완성 후보가 비어 있다**
가중치를 못 받았다. `.\.venv\Scripts\python.exe scripts\fetch_weights.py` 를 다시 돌린다.
회사망 프록시로 막히면, 스크립트가 출력하는 URL 로 직접 받아 `bundle/weights/` 에 넣으면 된다.
후보는 제안일 뿐이라 `.pt` 경로를 직접 입력해도 된다.

**화면이 비어 있다 / 404 가 뜬다**
`frontend/dist` 가 없다. 빌드 산출물이라 저장소에 들어 있지 않다.
`cd frontend; npm install; npm run build` 로 만든 뒤 다시 실행한다.

**GPU 목록이 비어 있다**
`nvidia-smi` 가 PATH 에 없다. PowerShell 에서 `nvidia-smi` 를 쳐서 확인한다.
안 되면 NVIDIA 드라이버를 다시 설치한다. GPU 없이 CPU 로 학습하려면 GPU 선택을 비워둔다.

**학습이 시작되자마자 CPU 로 돈다**
CUDA 없는 torch 가 깔렸다. 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.11.0+cu128 True` 가 나와야 한다. `+cpu` 면 다시 깐다:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**포트 8000 이 이미 쓰이고 있다**
점유 중인 프로세스를 찾는다:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
```

이 콘솔을 이미 띄워둔 것일 수 있다. 포트를 바꾸려면 `backend/app/core/config.py` 의 `PORT` 를 고친다.

**`.ps1` 실행이 차단된다**
위 [스크립트 실행이 막힐 때](#스크립트-실행이-막힐-때) 참고.

---

## 구조

```
backend/
  app/api/         REST + WebSocket 엔드포인트
  app/core/        설정 · SQLite
  app/services/    데이터셋 등록 · 모델 경로 해석 · 파라미터 스키마 · 실행 큐 ·
                   이벤트 tail · 추론 · GPU 조회
  hooks/           sitecustomize.py + yoloweb_events.py (ultralytics 콜백)
  train_worker.py  학습 워커 (독립 프로세스)
  export_worker.py 내보내기 워커 (독립 프로세스)
frontend/          React + Vite + TS
scripts/           setup.ps1 · start.ps1 · fetch_weights.py · relocate.py · build_bundle.ps1
bundle/            단독망 반입 패키지 (install.ps1 · 설치안내.md 만 커밋)
storage/           datasets/ · runs/ 산출물 (실행하면서 생긴다)
app.db             SQLite 메타 (실행하면서 생긴다)
```

### 개발용 실행

백엔드만 띄우려면:

```powershell
.\.venv\Scripts\python.exe backend\run.py
```

프론트엔드를 고칠 때는 개발 서버를 따로 띄운다:

```powershell
cd frontend
npm run dev
```

`http://localhost:5173` 이 `/api` 를 백엔드로 프록시한다. 배포 시에는 `npm run build` 로
`frontend/dist` 를 만들면 백엔드가 그대로 서빙한다.

### 학습 진행 상황이 전달되는 경로

학습은 웹서버와 분리된 별도 프로세스에서 돈다. 워커는 진행 상황을
`storage/runs/<run_id>/events.jsonl` 에 한 줄씩 append 하고, 백엔드는 그 파일을 tail 해
WebSocket 으로 브라우저에 흘린다.

파일이 단일 진실 원천이라 **브라우저를 새로고침해도, 백엔드를 재시작해도 차트가 완전히 복원된다.**
백엔드가 재시작되면 살아 있는 워커를 PID 로 다시 붙잡아 학습이 그대로 이어진다.

### 콜백을 전역으로 등록하는 이유

멀티 GPU(DDP) 학습에서 ultralytics 는 임시 `.py` 파일로 트레이너를 새로 조립해
자식 프로세스에서 실행한다(`ultralytics/utils/dist.py`). 이때 `model.add_callback()` 으로 붙인
콜백은 자식에게 전달되지 않는다. 그래서 워커는 `PYTHONPATH` 에 `backend/hooks` 를 넣고,
`sitecustomize.py` 가 모든 인터프리터 기동 시점에 콜백을 전역 등록한다.

`YOLOWEB_RUN_DIR` 환경변수가 없으면 훅은 아무 일도 하지 않으므로, 이 PC 의 다른 Python 실행에는
영향이 없다.

## 오프라인(단독망) 운영

`backend/app/core/config.py` 가 `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false` 를 세운다.
이게 없으면 ultralytics 가 import 시점에 DNS 를 조회해 기동이 지연된다.

### 반입 패키지 만들기

대상 PC 가 **파이썬도 CUDA 도 없는 순정 Windows** 라는 전제로, 런타임까지 통째로 반입한다.
인터넷이 되는 PC 에서:

```powershell
.\scripts\fetch_weights.py     # 가중치 (setup.ps1 이 이미 돌렸다면 생략)
.\scripts\build_bundle.ps1
```

`bundle/` 이 다음처럼 채워진다 (약 3 GB).

| 폴더 | 내용 |
|---|---|
| `prereq/VC_redist.x64.exe` | **필수.** 없으면 대상 PC 에서 `import torch` 가 실패한다 |
| `prereq/python-3.11.9-amd64.exe` | Python 오프라인 설치 파일 |
| `wheels/` | 모든 의존성 휠 (torch cu128 포함) |
| `weights/`, `fonts/`, `frontend-dist/` | 가중치 · 폰트 · 미리 빌드한 화면 |
| `requirements.lock.txt`, `install.ps1` | 설치 목록과 설치 스크립트 |

대상 PC 에서 관리자 PowerShell 로 `bundle\install.ps1` 을 실행한다.
자세한 절차와 주의사항은 [bundle/설치안내.md](bundle/설치안내.md).

**CUDA Toolkit 은 필요 없다** — torch 휠이 CUDA 런타임 DLL 을 담고 있다.
다만 **NVIDIA 그래픽 드라이버는 pip 로 배포할 수 없어** 직접 반입해야 한다.

빌드 스크립트에서 지키는 두 가지:

- 의존성을 **한 번의 해석으로** 받는다. torch 를 따로 받고 나머지를 따로 받으면
  ultralytics 의 torch 의존성이 PyPI 기준으로 다시 풀려 **CPU 빌드 torch 가 추가로 딸려 들어온다.**
  받은 뒤 같은 패키지의 휠이 둘 이상이면 빌드를 중단한다.
- `requirements.txt` 는 **ASCII 로만** 쓴다. pip 는 요구사항 파일을 로케일 인코딩(한국어 Windows 는 cp949)으로
  읽어서, 한글 주석이 있으면 대상 PC 의 `pip install -r` 이 그대로 깨진다.

모델 입력창의 자동완성은 `bundle/weights/` 에 **실제로 있는 파일**과 이전 학습의 `best.pt` 를 제안한다.

## 모델 지정

모델은 **경로로 지정한다.** 고정된 드롭다운 목록이 없다.

- `bundle/weights/` 의 `.pt`, **이전 학습의 `best.pt`**(이어서 학습), `yolo11n.yaml`(처음부터 학습)이
  자동완성 후보로 뜬다. 목록에 없는 경로도 직접 입력할 수 있다.
- 입력하면 서버가 즉시 검증해 초록/빨강으로 알려준다. 틀린 경로면 학습 시작 버튼이 잠긴다.
- 고른 가중치는 run 폴더의 `inputs/` 로 **복사**된다. 큐에서 기다리는 동안 원본이 지워져도 안전하고,
  나중에 "이 run 이 무슨 가중치로 시작했는지"가 run 폴더 안에 남는다.

경로 해석은 `backend/app/services/models.py` 한 곳에서만 한다.
워커의 작업 디렉터리가 `bundle/weights` 라서, 상대 경로는 API 프로세스와 워커에서 다르게 풀린다.

## 데이터셋 등록

두 가지 경로가 같은 파이프라인을 탄다.

- **zip 업로드** — 안전 해제 후 `storage/datasets/<id>/data/` 에 풀린다
- **로컬 경로 지정** — 복사하지 않고 원본을 그대로 참조한다. 원본에는 아무것도 쓰지 않는다

이후 구조 자동 감지 → train/val 이 없으면 자동 분할 → `data.yaml` 생성 → 검수 리포트 작성.

zip 해제는 경로 탈출(`../`)·절대 경로·심볼릭 링크 항목을 거부하고, 확장자 화이트리스트와
용량·항목 수 상한을 적용한다.

### 검수 화면

"라벨 없는 이미지 37건" 같은 **숫자만으로는 판단할 수 없다.** 실제로 열어봐야 라벨링 실수인지
배경 이미지인지 안다. 그래서 문제 유형별로 그 이미지들만 볼 수 있게 해 둔다.

카테고리는 안정된 코드로 구분한다 — `missing_label` / `empty_label` / `coord_out_of_range`(값이 0~1 밖) /
`box_out_of_image`(값은 정상인데 박스가 이미지를 벗어남) / `invalid_class` / `malformed_line` / `orphan_label`.
카테고리마다 최대 2,000건까지 기록하고, 잘렸으면 **"37,412건 중 2,000건 표시"** 라고 밝힌다.

박스 크기·종횡비 분포도 함께 보여준다. 작은 객체가 대부분인데 `imgsz` 를 작게 잡으면 아예 안 잡히므로,
파라미터를 고르기 전에 알아야 하는 정보다. 그래서 **새 학습 화면에서도** 데이터셋 검수를 펼쳐 볼 수 있다.

## 추론 테스트

학습된 가중치로 임의 이미지에 추론해 박스와 검출 표를 본다.

추론은 **서버가 CPU 로 고정한다.** 요청으로 GPU 를 고를 수 있게 두면 학습 중인 GPU 에 추론이 올라가
학습이 OOM 으로 죽는다. 모델은 `(경로, mtime, size)` 로 캐시하고, 캐시 교체와 추론은 세마포어로 직렬화한다.

## 내보내기

`best.pt` 를 ONNX / TorchScript / TensorRT 로 변환한다. 오래 걸리므로 별도 프로세스로 띄우고 폴링으로 확인한다.

TensorRT 는 GPU 를 쓰므로 **학습과 같은 GPU 슬롯을 두고 경쟁한다.** 프로세스를 분리해도 VRAM 은 분리되지
않기 때문에, 학습이 그 GPU 를 쓰는 중이면 요청을 거절한다. 같은 run 에 대한 중복 요청도 409 로 막는다.

## 정지

- **안전 정지** — `stop.flag` 를 만들어 콜백이 `trainer.stop` 을 세운다. 에폭 경계에서 정상 종료되어
  `best.pt` 와 플롯이 보존된다. DDP 에서도 rank 간 전파를 타므로 안전하다.
- **강제 종료** — 프로세스 트리를 kill 한다. 산출물 보존은 보장되지 않는다.

## 보안 범위

서버는 `127.0.0.1` 에만 바인딩되고 **인증이 없다.** 단독망 단일 PC 운영을 전제로 한 설계다.
다른 PC 에서 접근할 수 있게 열려면 인증을 먼저 붙여야 한다.

## 라이선스

미정. 이 프로젝트는 [Ultralytics](https://github.com/ultralytics/ultralytics) 에 의존하며
Ultralytics 는 AGPL-3.0 이다. 배포·재사용 전에 확인이 필요하다.
