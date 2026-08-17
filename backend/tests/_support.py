"""테스트가 backend 패키지를 import 할 수 있게 경로를 잡고, 임시 저장소를 만들어 준다.

여기서 임시 경로를 쓰는 이유: config 의 경로는 모듈 상수라 사용자의 실제 app.db 와
storage/ 를 그대로 가리킨다. 테스트가 그걸 건드리면 안 된다.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def isolate_storage() -> Path:
    """DB 와 runs 폴더를 임시 경로로 돌린다. 임시 루트를 돌려준다."""
    root = Path(tempfile.mkdtemp(prefix="yoloweb_test_"))

    from app.core import config, db
    from app.services import dataset_ingest, jobs, run_manager

    config.DB_PATH = root / "app.db"
    config.RUNS_DIR = root / "runs"
    config.DATASETS_DIR = root / "datasets"
    config.STORAGE_DIR = root
    db.DB_PATH = config.DB_PATH
    db._conn = None  # 이전 테스트가 열어 둔 연결을 버린다
    run_manager.RUNS_DIR = config.RUNS_DIR

    # 각 모듈은 import 시점에 config 의 값을 자기 이름으로 묶어 둔다.
    # config 만 바꾸면 그 바인딩은 실제 storage/ 를 계속 가리킨다.
    jobs.RUNS_DIR = config.RUNS_DIR
    jobs.DATASETS_DIR = config.DATASETS_DIR
    jobs.OWNER_ROOTS = {"run": config.RUNS_DIR, "dataset": config.DATASETS_DIR}
    dataset_ingest.DATASETS_DIR = config.DATASETS_DIR

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return root


def isolate_weights(test, names: list[str]) -> Path:
    """bundle/weights 를 임시 폴더로 돌리고 주어진 이름의 빈 .pt 를 만든다.

    isolate_storage 는 이걸 하지 않는다 — 그래서 지금 테스트들은 이 PC 의 진짜
    bundle/weights 를 읽는다. 모델을 고르는 코드를 검증하려면 그 폴더의 내용이
    테스트가 정한 것이어야 한다.

    param_schema 와 models 는 각자 import 시점에 WEIGHTS_DIR 을 자기 이름으로
    묶어 두므로 **두 모듈 모두** 갈아끼워야 한다(isolate_storage 의 jobs.RUNS_DIR 과 같은 이유).

    **jobs 와 run_manager 도 WEIGHTS_DIR 을 묶어 두지만 여기서는 건드리지 않는다.** 그쪽은
    워커 subprocess 의 cwd 로 쓰므로, 워커를 띄우는 테스트를 새로 쓸 때는 그 둘도 함께
    갈아끼워야 한다 — 안 그러면 조용히 진짜 bundle/weights 를 쓴다.
    """
    from app.services import models, param_schema

    root = Path(tempfile.mkdtemp(prefix="yoloweb_weights_"))
    for name in names:
        (root / name).write_bytes(b"")

    originals = [(param_schema, param_schema.WEIGHTS_DIR), (models, models.WEIGHTS_DIR)]
    for module, old in originals:
        setattr(module, "WEIGHTS_DIR", root)
        test.addCleanup(setattr, module, "WEIGHTS_DIR", old)
    return root


def force_worker_alive(test, alive: bool = True) -> None:
    """procs.is_our_worker 를 잠시 바꾼다. 테스트가 끝나면 원래대로 돌린다.

    직접 되돌리려 하면 틀리기 쉽다 — jobs.procs 는 app.core.procs 와 같은 모듈 객체라,
    패치한 뒤에 그 모듈에서 다시 읽어 오면 방금 넣은 가짜를 자기 자신에 덮어쓴다.
    """
    from app.core import procs

    original = procs.is_our_worker
    procs.is_our_worker = lambda pid, started_at: alive
    test.addCleanup(lambda: setattr(procs, "is_our_worker", original))
