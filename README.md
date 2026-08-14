# YOLO 학습 웹 콘솔

브라우저에서 데이터셋을 등록하고, 하이퍼파라미터를 폼으로 조정하고,
학습 진행·지표·로그·에폭별 예측 이미지를 실시간으로 보는 콘솔.

인터넷이 없는 단독망 운영을 전제로 만들었다. 실행 중에는 외부 요청을 하지 않는다.

이 저장소에는 **예제 데이터셋과 기존 학습 결과 6건이 함께 들어 있다.** 세팅만 하면
학습을 돌리기 전에도 차트·로그·에폭별 예측 이미지를 바로 볼 수 있다.

---

## 요구사항

| | |
|---|---|
| OS | Windows 10 / 11 (스크립트가 PowerShell 기준) |
| Python | **3.11** 권장. [3.11.9 내려받기](https://www.python.org/downloads/release/python-3119/) — 설치할 때 **"Add python.exe to PATH"** 체크 |
| GPU | NVIDIA GPU + 최신 드라이버. 없어도 `-Cpu` 로 돌아가지만 학습이 매우 느리다 |
| Node.js | **필요 없다.** 빌드된 `frontend/dist` 가 저장소에 들어 있다. 프론트엔드 소스를 고칠 때만 필요 |
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
사전학습 가중치 다운로드 → 저장된 경로를 이 PC 에 맞게 수정 → (Node 가 있으면) 프론트엔드 빌드.
torch 가 커서 처음 한 번은 몇 분 걸린다.

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

## 처음 5분: 예제 따라하기

**1. 기존 학습 결과부터 본다.**
접속하면 왼쪽 실행 목록에 6건이 이미 있다. 아무거나 눌러보면 손실·mAP 차트, 학습 로그,
에폭별 예측 이미지, 혼동 행렬이 그대로 복원된다. 학습을 돌리지 않아도 화면이 어떻게 도는지
전부 확인할 수 있다.

| 실행 이름 | 상태 | 무엇을 보나 |
|---|---|---|
| `smoke-3ep` | 완료 | 가장 기본. 3 에폭 완주 |
| `epoch-preview-test` | 완료 | 에폭별 예측 이미지 슬라이더 |
| `실시간 차트 확인` | 완료 | 에폭이 많아 차트가 제일 잘 보인다 |
| `정지 테스트` | 정지됨 | 안전 정지로 끝낸 실행 (`best.pt` 보존) |
| `재시작 생존 테스트` | 정지됨 | 백엔드를 재시작해도 이어지는지 본 실행 |
| `리뷰 반영 후 검증` | 완료 | 아래 3번과 같은 설정으로 돌린 것 |

**2. 예제 데이터셋을 등록한다.**
**데이터셋 추가 → 로컬 경로**에 아래를 넣는다 (`<clone 경로>` 는 실제 경로로).

```
<clone 경로>\examples\sample_dataset
```

이미지 16장 / 클래스 2개(red, blue) 짜리 합성 데이터다. train/val 이 없어서 자동으로
12 : 4 로 나뉘고, 검수 리포트가 뜬다. 자세한 건 [examples/README.md](examples/README.md).

**3. 학습을 돌린다.**
**새 학습**에서 모델 `yolo11n.pt` / 에폭 3 / 이미지 크기 160 / 배치 8. GPU 면 1분 안에 끝난다.
차트가 에폭마다 갱신되고 로그가 실시간으로 흐르면 정상이다.
`리뷰 반영 후 검증` 실행이 정확히 이 설정이라 결과를 비교해 볼 수 있다.

---

## 저장소에 들어 있는 것 / 없는 것

| | 포함 | 비고 |
|---|---|---|
| 백엔드·프론트엔드 소스 | O | |
| `frontend/dist` (빌드 결과) | O | Node.js 없이 바로 실행하려고 커밋해 둔다 |
| `examples/sample_dataset` | O | 등록해 볼 원본 데이터 (약 27 KB) |
| `storage/datasets`, `storage/runs` | O | 이미 등록된 데이터셋 + 학습 결과 6건 (약 70 MB) |
| `app.db` | O | **이게 있어야 실행 목록이 화면에 뜬다** |
| `bundle/weights/*.pt` | X | `scripts/fetch_weights.py` 가 받는다 (setup 이 자동 실행) |
| `node_modules`, `.venv` | X | setup 이 만든다 |

프론트엔드 소스를 고쳤다면 `npm run build` 로 `frontend/dist` 를 다시 만들고
**`dist` 도 함께 커밋**해야 다른 PC 에 반영된다.

## 다른 경로에 clone 했다면

`storage/` 안의 데이터셋 메타와 실행 설정에는 만들어질 당시의 절대경로가 들어 있다.
경로가 달라지면 이걸 고쳐야 한다.

```powershell
.\.venv\Scripts\python.exe scripts\relocate.py
```

`setup.ps1` 이 이미 한 번 돌려주므로 보통은 따로 실행할 일이 없다.
저장소 폴더를 나중에 옮겼을 때만 다시 돌리면 된다. 같은 경로에서 여러 번 돌려도 안전하다.

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

**모델 드롭다운이 비어 있다 / 전부 `(미반입)` 이다**
가중치를 못 받았다. `.\.venv\Scripts\python.exe scripts\fetch_weights.py` 를 다시 돌린다.
회사망 프록시로 막히면, 스크립트가 출력하는 URL 로 직접 받아 `bundle/weights/` 에 넣으면 된다.

**기존 학습 결과 6건이 목록에 안 뜬다**
`app.db` 가 없거나 clone 이 덜 됐다. `git status` 로 확인한다.
목록에는 뜨는데 데이터셋이 "없음"으로 나오면 `scripts/relocate.py` 를 돌린다.

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
  app/api/        REST + WebSocket 엔드포인트
  app/core/       설정 · SQLite
  app/services/   데이터셋 등록 · 파라미터 스키마 · 실행 큐 · 이벤트 tail · GPU 조회
  hooks/          sitecustomize.py + yoloweb_events.py (ultralytics 콜백)
  train_worker.py 학습 워커 (독립 프로세스)
frontend/         React + Vite + TS
scripts/          setup.ps1 · start.ps1 · fetch_weights.py · relocate.py
examples/         예제 데이터셋
bundle/weights/   사전학습 가중치 (.pt) — 여기 있는 파일만 모델 목록에 뜬다
storage/          datasets/ · runs/ 산출물
app.db            SQLite 메타
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

반입 전에 준비할 것:

- `bundle/weights/` — 쓸 사전학습 가중치(`yolo11n.pt` 등)와 AMP 체크용 `yolo26n.pt`
- `bundle/wheels/` — `pip download` 로 받은 휠 (torch 포함)
- `frontend/dist` — 미리 빌드한 정적 파일 (단독망에서 `npm install` 하지 않는다)

인터넷이 되는 PC 에서 `scripts/setup.ps1` 을 한 번 돌린 뒤 폴더째 옮기면 위 세 가지가 모두 채워진다.
옮긴 곳의 경로가 다르면 `scripts/relocate.py` 를 돌린다.

모델 드롭다운은 `bundle/weights/` 에 **실제로 있는 파일**을 `(번들)` 로, 없는 표준 모델을
`(미반입)` 으로 구분해 보여준다.

## 데이터셋 등록

두 가지 경로가 같은 파이프라인을 탄다.

- **zip 업로드** — 안전 해제 후 `storage/datasets/<id>/data/` 에 풀린다
- **로컬 경로 지정** — 복사하지 않고 원본을 그대로 참조한다. 원본에는 아무것도 쓰지 않는다

이후 구조 자동 감지 → train/val 이 없으면 자동 분할 → `data.yaml` 생성 → 검수 리포트 작성.

zip 해제는 경로 탈출(`../`)·절대 경로·심볼릭 링크 항목을 거부하고, 확장자 화이트리스트와
용량·항목 수 상한을 적용한다.

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
