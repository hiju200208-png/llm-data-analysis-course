
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parents[1]

NAVER_PATH = BASE_DIR / "data" / "processed" / "naver_trend.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 2. NAVER 데이터 읽기
naver = pd.read_csv(NAVER_PATH)

naver["period"] = pd.to_datetime(naver["period"])
naver["ratio"] = pd.to_numeric(naver["ratio"])

# 3. 키워드별 최근 데이터 확인
latest = (
    naver.sort_values("period")
    .groupby("keyword")
    .tail(1)
    .sort_values("ratio", ascending=False)
)

print("=== NAVER 최근 검색 관심도 ===")
print(latest.to_string(index=False))

# 4. 그래프 생성
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(10, 5))

for keyword, group in naver.groupby("keyword"):
    group = group.sort_values("period")

    plt.plot(
        group["period"],
        group["ratio"],
        marker="o",
        label=keyword,
    )

plt.title("NAVER 검색 관심도 추이")
plt.xlabel("기간")
plt.ylabel("상대 관심도")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# 5. 그래프 저장
output_path = REPORT_DIR / "naver_trend.png"

plt.savefig(output_path, dpi=150)
plt.close()

print("\n[PASS] NAVER 그래프 생성")
print("저장 위치:", output_path)



# ============================================================
# STEP 13. 분석 요약 CSV 생성
# ============================================================

# 키워드별 최근 2개 기간의 데이터 추출
naver_sorted = naver.sort_values(["keyword", "period"])

summary_rows = []

for keyword, group in naver_sorted.groupby("keyword"):

    group = group.sort_values("period")

    latest_row = group.iloc[-1]

    previous_ratio = (
        float(group.iloc[-2]["ratio"])
        if len(group) >= 2
        else None
    )

    latest_ratio = float(latest_row["ratio"])

    change = (
        latest_ratio - previous_ratio
        if previous_ratio is not None
        else None
    )

    summary_rows.append({
        "category": "NAVER",
        "keyword": keyword,
        "period": latest_row["period"].strftime("%Y-%m"),
        "ratio": latest_ratio,
        "previous_ratio": previous_ratio,
        "change": change,
    })

# DataFrame 생성
summary_df = pd.DataFrame(summary_rows)

# CSV 저장
summary_path = REPORT_DIR / "analysis_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)

print("\n[PASS] 분석 요약 CSV 생성")
print("saved:", summary_path)
print(summary_df.to_string(index=False))
