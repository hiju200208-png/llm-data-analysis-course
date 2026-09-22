import streamlit as st
import pandas as pd

from kiwipiepy import Kiwi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. 데이터 불러오기
# =========================
@st.cache_data
def load_data():
    return pd.read_csv(
        "books_improved.csv",
        encoding="utf-8-sig"
    )


df = load_data()


# =========================
# 2. 형태소 분석 + 전처리
# =========================
kiwi = Kiwi()

TARGET_TAGS = {"NNG", "NNP", "SL"}

stopwords = {
    "에디션",
}


def tokenize_title(text):
    tokens = kiwi.tokenize(str(text))

    words = [
        token.form
        for token in tokens
        if token.tag in TARGET_TAGS
    ]

    return " ".join(words)


def clean_tokens(text):
    words = str(text).split()

    cleaned = [
        word
        for word in words
        if len(word) >= 2
        and not word.isdigit()
        and word not in stopwords
    ]

    return " ".join(cleaned)


def preprocess_title(text):
    tokenized = tokenize_title(text)
    cleaned = clean_tokens(tokenized)

    return cleaned


# =========================
# 3. Improved 분류 모델 학습
# =========================
X = df["상품명"]
y = df["분야"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 훈련 데이터에만 전처리 적용
train_texts = X_train.apply(preprocess_title)

improved_vectorizer = TfidfVectorizer()

X_train_improved = improved_vectorizer.fit_transform(
    train_texts
)

improved_model = MultinomialNB()

improved_model.fit(
    X_train_improved,
    y_train
)


# =========================
# 4. 추천 함수
# =========================
def recommend_books(selected_index, top_n=5):

    selected_book = df.loc[selected_index]
    selected_category = selected_book["분야"]

    # 같은 분야만 후보
    candidate_df = df[
        df["분야"] == selected_category
    ].copy()

    # 형태소 분석 + 단어 필터링
    candidate_df["추천_텍스트"] = (
        candidate_df["상품명"]
        .apply(preprocess_title)
    )

    # TF-IDF
    recommendation_vectorizer = TfidfVectorizer()

    tfidf_matrix = (
        recommendation_vectorizer.fit_transform(
            candidate_df["추천_텍스트"]
        )
    )

    # 선택한 책의 위치
    selected_position = (
        candidate_df.index.get_loc(selected_index)
    )

    # Cosine Similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_position],
        tfidf_matrix,
    ).flatten()

    candidate_df["similarity"] = similarity_scores

    # 자기 자신 제외 + 유사도 0 제외
    recommendations = candidate_df[
        (candidate_df.index != selected_index)
        & (candidate_df["similarity"] > 0)
    ]

    # 높은 순
    recommendations = recommendations.sort_values(
        "similarity",
        ascending=False,
    ).head(top_n)

    return recommendations


# =========================
# 5. Streamlit 화면
# =========================
st.title("📚 도서 분야 예측 & 추천")

menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "도서 분야 예측",
        "비슷한 도서 추천",
    ],
)


# =========================
# 메뉴 1. 도서 분야 예측
# =========================
if menu == "도서 분야 예측":

    st.header("📖 도서 분야 예측")

    user_title = st.text_input(
        "도서 제목을 입력하세요"
    )

    if st.button("분야 예측"):

        if user_title.strip():

            # 새 제목 전처리
            processed_title = preprocess_title(
                user_title
            )

            # 중요!
            # 새 데이터이므로 fit_transform이 아니라 transform
            title_vector = (
                improved_vectorizer.transform(
                    [processed_title]
                )
            )

            prediction = improved_model.predict(
                title_vector
            )[0]

            st.write(
                "전처리 결과:",
                processed_title
            )

            st.success(
                f"예상 분야: {prediction}"
            )

        else:
            st.warning(
                "도서 제목을 입력해주세요."
            )


# =========================
# 메뉴 2. 비슷한 도서 추천
# =========================
elif menu == "비슷한 도서 추천":

    st.header("🔍 비슷한 도서 추천")

    selected_index = st.selectbox(
        "도서를 선택하세요",
        options=df.index,
        format_func=lambda x: df.loc[x, "상품명"],
    )

    selected_book = df.loc[selected_index]

    st.write(
        "선택 도서:",
        selected_book["상품명"]
    )

    st.write(
        "분야:",
        selected_book["분야"]
    )

    if st.button("추천하기"):

        recommendations = recommend_books(
            selected_index,
            top_n=5,
        )

        if recommendations.empty:

            st.info(
                "현재 기준으로 유사도가 있는 "
                "추천 도서를 찾지 못했습니다."
            )

        else:

            st.subheader("추천 결과")

            result = recommendations[
                [
                    "상품명",
                    "분야",
                    "similarity",
                ]
            ].copy()

            result["similarity"] = (
                result["similarity"].round(4)
            )

            st.dataframe(
                result,
                hide_index=True,
                use_container_width=True,
            )