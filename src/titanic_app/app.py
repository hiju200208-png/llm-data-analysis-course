"""Titanic 생존 예측 Streamlit 서비스.

Notebook(notebooks/ch05_titanic/00.ipynb)에서 확인한 전처리·예측 순서를 그대로 재사용합니다.

사용자 입력
→ FamilySize / IsAlone 생성
→ 숫자형 결측치 처리
→ 범주형 결측치 처리
→ One-Hot Encoding
→ StandardScaler 표준화
→ 숫자형 + 범주형 결합
→ 저장된 모델 예측

실행:
    streamlit run src/titanic_app/app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "titanic_model_bundle.joblib"
CONTRACT_PATH = MODELS_DIR / "titanic_model_contract.json"


@st.cache_resource
def load_bundle_and_contract():
    bundle = joblib.load(BUNDLE_PATH)
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        contract = json.load(f)
    return bundle, contract


def build_passenger_frame(pclass, sex, age, sibsp, parch, fare, embarked):
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    }])
    passenger["FamilySize"] = passenger["SibSp"] + passenger["Parch"] + 1
    passenger["IsAlone"] = (passenger["FamilySize"] == 1).astype(int)
    return passenger


def predict(bundle, contract, passenger: pd.DataFrame):
    numeric_features = contract["numeric_features"]
    categorical_features = contract["categorical_features"]

    num_imputed = bundle["numeric_imputer"].transform(passenger[numeric_features])
    num_scaled = bundle["scaler"].transform(num_imputed)

    cat_imputed = bundle["categorical_imputer"].transform(passenger[categorical_features])
    cat_encoded = bundle["encoder"].transform(cat_imputed)

    ready = pd.DataFrame(
        np.hstack([num_scaled, cat_encoded]),
        columns=contract["prepared_feature_columns"],
    )

    model = bundle["model"]
    prediction = int(model.predict(ready)[0])
    positive_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(ready)[0][positive_index])
    return prediction, probability


def main():
    st.set_page_config(page_title="Titanic 생존 예측", page_icon="🚢")
    st.title("🚢 Titanic 생존 예측")
    st.caption("Notebook에서 학습한 전처리·모델을 그대로 불러와 새 승객의 생존 여부를 예측합니다.")

    if not BUNDLE_PATH.exists() or not CONTRACT_PATH.exists():
        st.error(
            "저장된 모델을 찾을 수 없습니다. "
            "notebooks/ch05_titanic/00.ipynb의 STEP 16까지 먼저 실행해 주세요."
        )
        return

    bundle, contract = load_bundle_and_contract()

    with st.form("passenger_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Pclass (객실 등급)", [1, 2, 3], index=2)
            sex = st.selectbox("Sex (성별)", ["male", "female"])
            age = st.number_input("Age (나이)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            fare = st.number_input("Fare (운임)", min_value=0.0, max_value=600.0, value=10.0, step=1.0)
        with col2:
            sibsp = st.number_input("SibSp (동승 형제/배우자 수)", min_value=0, max_value=10, value=0, step=1)
            parch = st.number_input("Parch (동승 부모/자녀 수)", min_value=0, max_value=10, value=0, step=1)
            embarked = st.selectbox("Embarked (탑승 항구)", ["S", "C", "Q"])

        submitted = st.form_submit_button("생존 여부 예측")

    if submitted:
        passenger = build_passenger_frame(pclass, sex, age, sibsp, parch, fare, embarked)
        prediction, probability = predict(bundle, contract, passenger)

        if prediction == 1:
            st.success(f"예측 결과: 생존 (Survived) — 생존 확률 {probability:.1%}")
        else:
            st.error(f"예측 결과: 비생존 (Not Survived) — 생존 확률 {probability:.1%}")

        st.subheader("모델 입력값")
        st.dataframe(passenger, hide_index=True)

    with st.expander("모델 정보"):
        st.write(f"모델: {contract['final_estimator']}")
        st.write(f"입력 Feature: {contract['raw_input_columns']}")
        st.write(f"파생 Feature: {contract['derived_features']}")
        st.write(f"스케일링: {contract['scaling']}")


if __name__ == "__main__":
    main()
