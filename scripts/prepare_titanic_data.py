"""Titanic 실습용 데이터를 내려받아 저장합니다.

실행 방법:
    python scripts/prepare_titanic_data.py

생성 위치:
    data/titanic/train.csv
"""

from pathlib import Path

import pandas as pd

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
TITANIC_DATA_DIR = Path("data/titanic")


def main() -> None:
    """Titanic train 데이터를 내려받아 data/titanic/train.csv로 저장합니다."""
    TITANIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TITANIC_DATA_DIR / "train.csv"

    df = pd.read_csv(TITANIC_URL)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Titanic 데이터 준비 완료")
    print(f"- {output_path}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
