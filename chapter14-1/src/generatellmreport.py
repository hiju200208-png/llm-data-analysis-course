
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# ============================================================
# STEP 1. 기본 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env", override=True)


# ============================================================
# STEP 2. 분석 데이터 불러오기
# ============================================================

def load_summary():

    summary_path = REPORTS_DIR / "analysis_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"분석 요약 파일이 없습니다: {summary_path}"
        )

    df = pd.read_csv(summary_path)

    if df.empty:
        raise RuntimeError(
            "분석 요약 데이터가 비어 있습니다."
        )

    # NaN 값을 JSON에서 사용할 수 있도록 변환
    df = df.astype(object).where(pd.notna(df), None)

    summary = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "results": df.to_dict(orient="records"),
    }

    print("[PASS] 분석 요약 데이터 로드")
    print("rows:", len(df))

    return summary


# ============================================================
# STEP 3. LLM 프롬프트 생성
# ============================================================

def build_prompt(summary):

    return f"""
당신은 데이터 분석 보고서 작성 보조자입니다.

아래 값은 Python에서 계산하고 사람이 기본 검증을
완료한 결과입니다.

{json.dumps(summary, ensure_ascii=False, indent=2)}

다음 형식으로 한국어 보고서를 작성하세요.

1. 분석 기준일
2. 검색 관심도 주요 변화
3. 지역 관광 수요 주요 특징
4. 관찰 가능한 패턴
5. 현재 데이터로 말할 수 없는 것
6. 추가 확인이 필요한 사항

중요한 작성 규칙:

- 검색 관심도는 실제 검색 건수가 아닌 상대지수라고
  반드시 명시하세요.

- 상관관계나 동시 변화만으로 인과관계를
  단정하지 마세요.

- 입력에 없는 사실이나 숫자를 새로 만들지 마세요.

- 관광 체류 강도와 관광 소비 강도는
  서로 다른 지표임을 명시하세요.

- 지역 데이터가 일부 지역에 한정되어 있다면
  전국적인 현상으로 일반화하지 마세요.

- 데이터가 부족한 항목은 추측하지 말고
  분석에 필요한 데이터가 부족하다고 설명하세요.

- 객관적이고 이해하기 쉬운 한국어로 작성하세요.

- Markdown 형식으로 보고서를 작성하세요.
"""


# ============================================================
# STEP 4. Gemini API 호출
# ============================================================

def call_gemini(prompt):

    from google import genai

    api_key = os.getenv("LLM_API_KEY")

    model = os.getenv("LLM_MODEL")

    if not api_key or not model:
        raise RuntimeError(
            "Gemini 설정이 없습니다. "
            ".env 파일의 LLM_API_KEY와 "
            "LLM_MODEL을 확인하세요."
        )

    client = genai.Client(api_key=api_key)

    print("[INFO] Gemini API 호출 중...")
    print("model:", model)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini 응답이 비어 있습니다."
        )

    print("[PASS] Gemini 응답 수신")

    return response.text


# ============================================================
# STEP 5. LLM 보고서 생성
# ============================================================

def generate_report(summary):

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    print("현재 LLM Provider:", repr(provider))

    prompt = build_prompt(summary)

    if provider == "gemini":

        text = call_gemini(prompt)

    else:

        raise RuntimeError(
            "현재 예제는 Gemini 기준입니다. "
            "LLM_PROVIDER=gemini로 설정하세요."
        )

    output_path = REPORTS_DIR / "llm_summary.md"

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    print("[PASS] LLM 보고서 저장 완료")
    print("saved:", output_path)

    return text


# ============================================================
# STEP 6. 전체 실행
# ============================================================

def main():

    print("=" * 50)
    print("STEP 16. LLM 분석 보고서 생성")
    print("=" * 50)

    # 1. 분석 요약 CSV 불러오기
    summary = load_summary()

    # 2. Gemini 보고서 생성
    report = generate_report(summary)

    # 3. 결과 출력
    print("\n" + "=" * 50)
    print("LLM 분석 보고서")
    print("=" * 50)

    print(report)

    print("\n[PASS] STEP 16 완료")


if __name__ == "__main__":
    main()
