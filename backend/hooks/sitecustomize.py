"""Python 이 모든 인터프리터 기동 시 자동으로 import 하는 훅.

학습 워커는 PYTHONPATH 에 이 디렉터리를 넣고 실행된다. torch.distributed.run 이 띄우는
DDP 자식 프로세스도 같은 환경변수를 물려받으므로 예외 없이 이 파일을 거친다.
그 덕분에 "모델에 붙인 콜백이 DDP 자식에게 전달되지 않는" ultralytics 의 구조를 우회한다.

YOLOWEB_RUN_DIR 이 없으면 아무 일도 하지 않는다 — 이 PC 의 다른 Python 실행에 영향이 없다.
"""

import os

if os.environ.get("YOLOWEB_RUN_DIR"):
    try:
        import yoloweb_events

        yoloweb_events.install()
    except Exception:  # 훅이 실패해도 학습 자체는 계속되어야 한다
        import traceback

        traceback.print_exc()
