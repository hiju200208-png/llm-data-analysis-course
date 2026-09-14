"""Titanic 모델링에 쓰이는 Feature 정의와 파생 Feature 생성 로직.

Notebook과 Streamlit 앱에서 동일한 Feature 정의를 재사용하기 위한 모듈입니다.
"""

import pandas as pd

RAW_INPUT_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
]

NUMERIC_FEATURES = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize",
]

CATEGORICAL_FEATURES = [
    "Pclass", "Sex", "Embarked", "IsAlone",
]

MODEL_FEATURE_COLUMNS = RAW_INPUT_COLUMNS + ["FamilySize", "IsAlone"]


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """FamilySize / IsAlone 파생 Feature를 추가합니다."""
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df
