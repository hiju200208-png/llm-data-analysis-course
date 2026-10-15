
import logging
import subprocess
import sys

from datetime import datetime
from pathlib import Path


# ============================================================
# STEP 1. 기본 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

log_path = (
    LOG_DIR
    / f"pipeline_{datetime.now():%Y%m%d}.log"
)


# ============================================================
# STEP 2. 로그 설정
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(
            log_path,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)


# ============================================================
# STEP 3. 실행할 Python 파일 목록
# ============================================================

STEPS = [
    "src/collect_naver.py",
    "src/collect_tourism.py",
    "src/analyze.py",
    "src/generatellmreport.py",
    "src/generate_pdf.py",
]


# ============================================================
# STEP 4. 개별 단계 실행
# ============================================================

def run_step(script):

    script_path = BASE_DIR / script

    # 파일 존재 여부 확인
    if not script_path.exists():
        logging.error(
            "파일이 존재하지 않습니다: %s",
            script,
        )
        raise SystemExit(1)

    logging.info("START %s", script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        text=True,
    )

    # 실행 실패 시 전체 파이프라인 중단
    if result.returncode != 0:
        logging.error(
            "FAIL %s (exit code: %s)",
            script,
            result.returncode,
        )
        raise SystemExit(result.returncode)

    logging.info("PASS %s", script)


# ============================================================
# STEP 5. 전체 파이프라인 실행
# ============================================================

def main():

    logging.info("=" * 50)
    logging.info("PIPELINE START")
    logging.info("=" * 50)

    for script in STEPS:
        run_step(script)

    logging.info("=" * 50)
    logging.info("PIPELINE SUCCESS")
    logging.info("=" * 50)

    logging.info("로그 저장 위치: %s", log_path)


if __name__ == "__main__":
    main()
