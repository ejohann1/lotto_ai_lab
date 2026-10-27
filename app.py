import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# Lotto AI Lab
# 통계 분석, 추천 조합, 백테스트
# ==========================================

APP_DIR = Path("lotto_data")
APP_DIR.mkdir(exist_ok=True)

CSV_PATH = APP_DIR / "lotto645.csv"
DB_PATH = APP_DIR / "lotto645.db"

BASE_URL = "https://www.dhlottery.co.kr/lt645"
HEADERS = {"User-Agent": "Mozilla/5.0"}

NUMBER_COLS = [f"n{i}" for i in range(1, 7)]
REQUIRED = ["round", "date"] + NUMBER_COLS + ["bonus"]
NUMBERS = np.arange(1, 46)

st.set_page_config(
    page_title="Lotto AI Lab",
    page_icon="🎱",
    layout="wide",
)

st.title("🎱 Lotto AI Lab")
st.caption("로또 6/45 통계 분석 · 번호 추천 · 백테스트")

st.info(
    "이 프로그램은 통계 학습 및 오락용입니다. "
    "과거 당첨 기록으로 미래 당첨을 보장할 수 없습니다."
)


# ==========================================
# 1. 데이터 수집
# ==========================================

@st.cache_resource
def get_session():
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)

    return session


def fetch_json(endpoint, params=None):
    response = get_session().get(
        f"{BASE_URL}/{endpoint}",
        params=params,
        timeout=(5, 25),
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("서버 응답 형식이 올바르지 않습니다.")

    return payload


def extract_items(payload):
    data = payload.get("data", {})

    if isinstance(data, dict):
        items = data.get("list")

        if isinstance(items, list) and items:
            return items

        if data.get("ltEpsd") is not None:
            return [data]

    raise ValueError(
        "당첨 데이터를 찾지 못했습니다. "
        "동행복권 API 응답 형식이 변경되었을 수 있습니다."
    )


def normalize_draw(item):
    round_no = int(item["ltEpsd"])
    date_raw = str(item["ltRflYmd"])

    nums = sorted(
        int(item[f"tm{i}WnNo"])
        for i in range(1, 7)
    )

    bonus = int(item["bnsWnNo"])

    if len(set(nums)) != 6:
        raise ValueError(f"{round_no}회 본번호 중복")

    if any(n < 1 or n > 45 for n in nums):
        raise ValueError(f"{round_no}회 번호 범위 오류")

    if bonus < 1 or bonus > 45 or bonus in nums:
        raise ValueError(f"{round_no}회 보너스 번호 오류")

    if len(date_raw) != 8 or not date_raw.isdigit():
        raise ValueError(f"{round_no}회 날짜 형식 오류")

    row = {
        "round": round_no,
        "date": (
            f"{date_raw[:4]}-{date_raw[4:6]}-"
            f"{date_raw[6:]}"
        ),
        "bonus": bonus,
    }

    for i, n in enumerate(nums, 1):
        row[f"n{i}"] = n

    return row


def validate_data(df):
    missing = [c for c in REQUIRED if c not in df.columns]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df[REQUIRED].copy()

    df["round"] = pd.to_numeric(
        df["round"], errors="raise"
    ).astype(int)

    for col in NUMBER_COLS + ["bonus"]:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        ).astype(int)

    if df["round"].duplicated().any():
        raise ValueError("중복 회차가 발견됐습니다.")

    for _, row in df.iterrows():
        nums = [int(row[c]) for c in NUMBER_COLS]
        bonus = int(row["bonus"])

        if len(set(nums)) != 6:
            raise ValueError(f"{row['round']}회 번호 중복")

        if any(n < 1 or n > 45 for n in nums):
            raise ValueError(f"{row['round']}회 번호 범위 오류")

        if bonus < 1 or bonus > 45 or bonus in nums:
            raise ValueError(f"{row['round']}회 보너스 오류")

    return df.sort_values("round").reset_index(drop=True)


def load_data():
    if not CSV_PATH.exists():
        return None

    return validate_data(pd.read_csv(CSV_PATH))


def save_data(df):
    df = validate_data(df)

    temp_path = CSV_PATH.with_suffix(".tmp")

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig",
    )

    temp_path.replace(CSV_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(
            "draws",
            conn,
            if_exists="replace",
            index=False,
        )

    return df


def get_latest_round():
    payload = fetch_json("selectPstLt645Info.do")
    items = extract_items(payload)

    return max(int(item["ltEpsd"]) for item in items)


def collect_data(progress=None):
    latest = get_latest_round()
    old = load_data()

    records = {}

    if old is not None:
        records = {
            int(row["round"]): row.to_dict()
            for _, row in old.iterrows()
        }

    missing = [
        r for r in range(1, latest + 1)
        if r not in records
    ]

    for idx, round_no in enumerate(missing, 1):
        payload = fetch_json(
            "selectPstLt645InfoNew.do",
            {
                "srchDir": "center",
                "srchLtEpsd": round_no,
            },
        )

        items = extract_items(payload)

        matches = [
            item for item in items
            if int(item["ltEpsd"]) == round_no
        ]

        if not matches:
            raise ValueError(
                f"{round_no}회 데이터를 찾지 못했습니다."
            )

        records[round_no] = normalize_draw(matches[0])

        if progress is not None:
            progress.progress(
                idx / max(len(missing), 1)
            )

        time.sleep(0.15)

    df = validate_data(pd.DataFrame(records.values()))

    expected = list(range(1, latest + 1))

    if df["round"].tolist() != expected:
        raise ValueError("회차 누락 또는 중복이 발견됐습니다.")

    save_data(df)

    return df, len(missing)


# ==========================================
# 2. 통계 계산
# ==========================================

def get_draws(df):
    return df[NUMBER_COLS].to_numpy(dtype=int)


def make_weights(draws, method):
    if method == "균등 무작위":
        return np.ones(45) / 45

    if method == "전체 출현 빈도":
        sample = draws
    else:
        sample = draws[-min(100, len(draws)):]

    counts = np.bincount(
        sample.ravel(),
        minlength=46,
    )[1:46].astype(float)

    # 라플라스 보정
    weights = counts + 1

    return weights / weights.sum()


def get_number_stats(draws):
    total_rounds = len(draws)

    counts = np.bincount(
        draws.ravel(),
        minlength=46,
    )[1:46]

    last_seen = {}

    for idx, draw in enumerate(draws):
        for number in draw:
            last_seen[int(number)] = idx

    return pd.DataFrame({
        "번호": NUMBERS,
        "출현 횟수": counts,
        "기대 출현 횟수": total_rounds * 6 / 45,
        "출현 빈도(%)": (
            counts / max(total_rounds, 1) * 100
        ),
        "미출현 기간": [
            total_rounds - 1 - last_seen.get(int(n), -1)
            for n in NUMBERS
        ],
    })


def get_features(draws):
    rows = []

    for draw in draws:
        nums = sorted(int(n) for n in draw)

        odd = sum(n % 2 for n in nums)
        low = sum(n <= 22 for n in nums)

        rows.append({
            "합계": sum(nums),
            "홀수": odd,
            "짝수": 6 - odd,
            "저번호(1~22)": low,
            "고번호(23~45)": 6 - low,
            "연속 번호 쌍": sum(
                nums[i + 1] - nums[i] == 1
                for i in range(5)
            ),
            "1~10": sum(1 <= n <= 10 for n in nums),
            "11~20": sum(11 <= n <= 20 for n in nums),
            "21~30": sum(21 <= n <= 30 for n in nums),
            "31~40": sum(31 <= n <= 40 for n in nums),
            "41~45": sum(41 <= n <= 45 for n in nums),
        })

    return pd.DataFrame(rows)


def bootstrap_ci(values, seed=1234, iterations=3000):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return tuple(
        np.quantile(means, [0.025, 0.975])
    )

# ==========================================
# 3. 추천 조합 생성
# ==========================================

def generate_combinations(
    weights,
    count,
    seed,
    sum_min=21,
    sum_max=255,
    odd_min=0,
    odd_max=6,
    low_min=0,
    low_max=6,
    max_consecutive=5,
    draws=None,
    avoid_previous=False,
):
    rng = np.random.default_rng(seed)
    found = set()

    previous = set()

    if draws is not None and len(draws):
        previous = set(int(n) for n in draws[-1])

    # 조건이 지나치게 까다로우면 무한 반복하지 않도록 제한
    max_attempts = max(50000, count * 10000)

    for _ in range(max_attempts):
        nums = tuple(sorted(
            rng.choice(
                NUMBERS,
                size=6,
                replace=False,
                p=weights,
            ).tolist()
        ))

        if nums in found:
            continue

        total = sum(nums)
        odd = sum(n % 2 for n in nums)
        low = sum(n <= 22 for n in nums)

        consecutive = sum(
            nums[i + 1] - nums[i] == 1
            for i in range(5)
        )

        if not sum_min <= total <= sum_max:
            continue

        if not odd_min <= odd <= odd_max:
            continue

        if not low_min <= low <= low_max:
            continue

        if consecutive > max_consecutive:
            continue

        if avoid_previous and set(nums) & previous:
            continue

        found.add(nums)

        if len(found) >= count:
            break

    return sorted(found)


# ==========================================
# 4. 백테스트
# ==========================================

def simulate_ticket(weights, rng):
    return set(
        rng.choice(
            NUMBERS,
            size=6,
            replace=False,
            p=weights,
        ).tolist()
    )


def run_backtest(
    draws,
    strategy,
    games=5,
    seed=2026,
    warmup=100,
):
    if len(draws) <= warmup:
        raise ValueError(
            "백테스트를 실행할 만큼 데이터가 충분하지 않습니다."
        )

    rng = np.random.default_rng(seed)
    rows = []

    uniform = np.ones(45) / 45

    for i in range(warmup, len(draws)):
        # 대상 회차의 당첨 번호는 예측에 사용하지 않음
        history = draws[:i]
        actual = set(int(n) for n in draws[i])

        weights = make_weights(history, strategy)

        strategy_hits = []
        random_hits = []

        for _ in range(games):
            predicted = simulate_ticket(weights, rng)
            random_predicted = simulate_ticket(uniform, rng)

            strategy_hits.append(
                len(predicted & actual)
            )

            random_hits.append(
                len(random_predicted & actual)
            )

        strategy_mean = float(np.mean(strategy_hits))
        random_mean = float(np.mean(random_hits))

        rows.append({
            "대상 회차": i + 1,
            "전략 평균 일치": strategy_mean,
            "무작위 평균 일치": random_mean,
            "평균 차이": strategy_mean - random_mean,
        })

    return pd.DataFrame(rows)


# ==========================================
# 5. 데이터 품질 검사
# ==========================================

def check_data_quality(df):
    expected = (
        int(df["round"].max())
        - int(df["round"].min())
        + 1
    )

    missing_rounds = sorted(
        set(range(
            int(df["round"].min()),
            int(df["round"].max()) + 1,
        ))
        - set(df["round"].astype(int))
    )

    return {
        "저장 회차 수": len(df),
        "중복 회차 수": int(df["round"].duplicated().sum()),
        "최초 회차": int(df["round"].min()),
        "최신 회차": int(df["round"].max()),
        "예상 연속 회차 수": expected,
        "누락 회차 수": len(missing_rounds),
        "누락 회차 목록": missing_rounds,
    }


# ==========================================
# 6. 추천 조합 설명
# ==========================================

def explain_combination(nums):
    nums = sorted(int(n) for n in nums)

    odd = sum(n % 2 for n in nums)
    low = sum(n <= 22 for n in nums)

    total = sum(nums)

    consecutive = sum(
        nums[i + 1] - nums[i] == 1
        for i in range(5)
    )

    return {
        "번호": " · ".join(f"{n:02d}" for n in nums),
        "합계": total,
        "홀수": odd,
        "짝수": 6 - odd,
        "저번호(1~22)": low,
        "고번호(23~45)": 6 - low,
        "연속 번호 쌍": consecutive,
    }

# ==========================================
# 7. 화면: 데이터 관리
# ==========================================

with st.sidebar:
    st.header("📁 데이터 관리")

    st.write(f"저장 폴더: {APP_DIR}")

    if st.button(
        "최신 데이터 수집 및 갱신",
        use_container_width=True,
    ):
        try:
            progress = st.progress(0)

            with st.spinner("동행복권 데이터를 수집합니다..."):
                updated_df, added = collect_data(progress)

            st.success(
                f"갱신 완료! 신규 {added}회 데이터를 확인했습니다."
            )
            st.rerun()

        except Exception as exc:
            st.error(f"수집 실패: {exc}")

    st.divider()

    st.caption(
        "데이터는 CSV와 SQLite 파일로 저장합니다. "
        "자동 갱신 버튼은 직접 눌러야 합니다."
    )


# ==========================================
# 8. 데이터 불러오기
# ==========================================

try:
    df = load_data()

except Exception as exc:
    st.error(f"저장 데이터 검증 실패: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning(
        "저장된 당첨 데이터가 없습니다. "
        "왼쪽 메뉴에서 '최신 데이터 수집 및 갱신'을 눌러 주세요."
    )
    st.stop()

draws = get_draws(df)
stats = get_number_stats(draws)
features = get_features(draws)


# ==========================================
# 9. 요약 지표
# ==========================================

st.header("📊 데이터 요약")

c1, c2, c3 = st.columns(3)

c1.metric(
    "저장 회차",
    f"{len(df):,}회",
)

c2.metric(
    "최신 저장 회차",
    f"{int(df['round'].max()):,}회",
)

c3.metric(
    "누적 본번호 기록",
    f"{len(df) * 6:,}개",
)

st.caption(
    "저장 회차는 현재 컴퓨터에 보관된 데이터 기준입니다."
)


# ==========================================
# 10. 번호별 출현 통계
# ==========================================

st.header("1. 번호별 출현 통계")

with st.expander("이 화면은 무엇을 의미하나요?", expanded=True):
    st.markdown("""
    - **출현 횟수:** 해당 번호가 과거 본번호에 나온 횟수
    - **기대 출현 횟수:** 공정한 추첨을 가정한 이론적 평균
    - **출현 빈도(%):** 전체 회차 중 해당 번호가 나온 비율
    - **미출현 기간:** 마지막 출현 이후 지나간 회차 수

    과거에 자주 나왔거나 오랫동안 나오지 않았다는 이유만으로
    다음 회차의 당첨 확률이 높아지는 것은 아닙니다.
    """)

metric = st.selectbox(
    "그래프에서 확인할 지표",
    [
        "출현 횟수",
        "기대 출현 횟수",
        "출현 빈도(%)",
        "미출현 기간",
    ],
)

st.bar_chart(
    stats.set_index("번호")[[metric]]
)

left, right = st.columns(2)

with left:
    st.subheader("🔥 출현 상위 번호")

    st.dataframe(
        stats.nlargest(10, "출현 횟수"),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("⏳ 미출현 기간 상위 번호")

    st.dataframe(
        stats.nlargest(10, "미출현 기간"),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("45개 번호 전체 통계 보기"):
    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================
# 11. 홀짝 · 고저 · 합계 · 구간 분석
# ==========================================

st.header("2. 당첨 번호 패턴 분석")

st.caption(
    "고저 기준: 저번호 1~22, 고번호 23~45. "
    "구간은 1~10, 11~20, 21~30, 31~40, 41~45입니다."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["홀짝", "고저", "합계", "구간 분석"]
)

with tab1:
    st.subheader("홀수 개수별 출현 분포")

    odd_counts = (
        features["홀수"]
        .value_counts()
        .reindex(range(7), fill_value=0)
        .rename_axis("홀수 개수")
        .to_frame("회차 수")
    )

    st.bar_chart(odd_counts)

    st.caption(
        "예: 홀수 개수가 3이면 나머지 3개는 짝수입니다."
    )

with tab2:
    st.subheader("저번호 개수별 출현 분포")

    low_counts = (
        features["저번호(1~22)"]
        .value_counts()
        .reindex(range(7), fill_value=0)
        .rename_axis("저번호 개수")
        .to_frame("회차 수")
    )

    st.bar_chart(low_counts)

    st.caption(
        "저번호가 4개라면 고번호는 2개입니다."
    )

with tab3:
    st.subheader("회차별 당첨 번호 합계")

    st.line_chart(
        pd.DataFrame({
            "회차": df["round"].to_numpy(),
            "합계": features["합계"].to_numpy(),
        }).set_index("회차")
    )

    st.metric(
        "과거 평균 합계",
        f"{features['합계'].mean():.2f}",
    )

    st.caption(
        "평균 합계는 과거 결과의 요약입니다. "
        "다음 회차의 합계를 예측하는 값이 아닙니다."
    )

with tab4:
    range_cols = [
        "1~10",
        "11~20",
        "21~30",
        "31~40",
        "41~45",
    ]

    st.subheader("번호 구간별 평균 출현 개수")

    st.bar_chart(
        features[range_cols].mean().rename("회차당 평균")
    )

    st.caption(
        "예를 들어 1~10 구간의 평균이 1.3이라면 "
        "한 회차에서 해당 구간 번호가 평균 1.3개 나왔다는 뜻입니다."
    )

st.subheader("연속 번호 쌍 분석")

consecutive_counts = (
    features["연속 번호 쌍"]
    .value_counts()
    .sort_index()
    .rename_axis("연속 번호 쌍")
    .to_frame("회차 수")
)

st.bar_chart(consecutive_counts)

st.caption(
    "예: 12, 13, 14는 연속 번호 쌍이 2개입니다. "
    "연속 번호가 있다는 이유만으로 당첨 가능성이 낮아지는 것은 아닙니다."
)


# ==========================================
# 12. 최근 당첨 결과
# ==========================================

st.header("3. 최근 당첨 결과")

st.dataframe(
    df.tail(10).iloc[::-1],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "최근 10회 결과를 최신 회차부터 표시합니다."
)


# ==========================================
# 13. 추천 번호 생성 및 필터링
# ==========================================

st.header("4. 추천 조합 실험실")

st.markdown("""
각 필터는 생성할 조합의 형태를 제한합니다.

- **균등 무작위:** 모든 번호에 동일한 가중치를 적용합니다.
- **전체 출현 빈도:** 과거 출현 횟수에 따라 번호별 가중치를 적용합니다.
- **최근 100회 빈도:** 최근 100회의 출현 횟수에 따라 가중치를 적용합니다.

빈도 기반 추천은 통계 실험을 위한 방식이며,
당첨 확률이 향상된다는 보장은 없습니다.
""")

method = st.selectbox(
    "추천 알고리즘",
    [
        "균등 무작위",
        "전체 출현 빈도",
        "최근 100회 빈도",
    ],
)

num_games = st.slider(
    "생성할 게임 수",
    min_value=1,
    max_value=20,
    value=5,
)

seed = st.number_input(
    "난수 시드",
    min_value=0,
    max_value=2**31 - 1,
    value=42,
)

col1, col2 = st.columns(2)

with col1:
    sum_min = st.number_input(
        "번호 합계 최소",
        min_value=21,
        max_value=269,
        value=100,
    )

    odd_min = st.slider(
        "홀수 최소 개수",
        0, 6, 2,
    )

    low_min = st.slider(
        "저번호 최소 개수",
        0, 6, 2,
    )

with col2:
    sum_max = st.number_input(
        "번호 합계 최대",
        min_value=21,
        max_value=269,
        value=200,
    )

    odd_max = st.slider(
        "홀수 최대 개수",
        0, 6, 4,
    )

    low_max = st.slider(
        "저번호 최대 개수",
        0, 6, 4,
    )

max_consecutive = st.slider(
    "허용할 연속 번호 쌍 최대 개수",
    min_value=0,
    max_value=5,
    value=3,
)

avoid_previous = st.checkbox(
    "직전 회차의 본번호를 하나도 포함하지 않기",
    value=False,
)

if st.button("추천 번호 생성", type="primary"):
    if sum_min > sum_max:
        st.error("합계 최소값이 최대값보다 큽니다.")

    elif odd_min > odd_max:
        st.error("홀수 최소 개수가 최대 개수보다 큽니다.")

    elif low_min > low_max:
        st.error("저번호 최소 개수가 최대 개수보다 큽니다.")

    else:
        weights = make_weights(draws, method)

        combos = generate_combinations(
            weights=weights,
            count=int(num_games),
            seed=int(seed),
            sum_min=int(sum_min),
            sum_max=int(sum_max),
            odd_min=int(odd_min),
            odd_max=int(odd_max),
            low_min=int(low_min),
            low_max=int(low_max),
            max_consecutive=int(max_consecutive),
            draws=draws,
            avoid_previous=avoid_previous,
        )

        if len(combos) < num_games:
            st.warning(
                f"조건을 만족하는 조합 {len(combos)}개를 찾았습니다. "
                "조건이 너무 엄격하면 필터를 완화해 주세요."
            )

        if combos:
            result_rows = [
                explain_combination(combo)
                for combo in combos
            ]

            result_df = pd.DataFrame(result_rows)
            result_df.insert(
                0,
                "게임",
                range(1, len(result_df) + 1),
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "추천 결과 CSV 다운로드",
                result_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="lotto_recommendations.csv",
                mime="text/csv",
            )


# ==========================================
# 14. 백테스트
# ==========================================

st.header("5. 백테스트 및 성능 검증")

st.markdown("""
백테스트는 과거 회차를 대상으로 추천 방식과 무작위 방식의
평균 일치 개수를 비교하는 실험입니다.

대상 회차의 당첨 번호는 그 회차를 예측할 때 사용하지 않습니다.
""")

st.caption(
    "평균 일치 개수는 1등 당첨 확률이 아니라 "
    "본번호 6개 중 몇 개가 일치했는지를 나타냅니다."
)

if len(draws) <= 100:
    st.warning(
        "현재 데이터는 100회 이하입니다. "
        "백테스트를 위해 101회 이상 데이터가 필요합니다."
    )

else:
    bt_method = st.selectbox(
        "백테스트 전략",
        [
            "전체 출현 빈도",
            "최근 100회 빈도",
        ],
        key="bt_method",
    )

    bt_games = st.slider(
        "회차당 실험 게임 수",
        1, 20, 5,
        key="bt_games",
    )

    warmup = st.number_input(
        "초기 학습 회차",
        min_value=50,
        max_value=len(draws) - 1,
        value=min(100, len(draws) - 1),
    )

    bt_seed = st.number_input(
        "백테스트 난수 시드",
        min_value=0,
        max_value=2**31 - 1,
        value=2026,
        key="bt_seed",
    )

    if st.button("백테스트 실행"):
        with st.spinner("백테스트를 실행하고 있습니다..."):
            result = run_backtest(
                draws,
                bt_method,
                games=int(bt_games),
                seed=int(bt_seed),
                warmup=int(warmup),
            )

        low_ci, high_ci = bootstrap_ci(
            result["평균 차이"],
            seed=int(bt_seed) + 1,
        )

        strategy_mean = result["전략 평균 일치"].mean()
        random_mean = result["무작위 평균 일치"].mean()
        diff_mean = result["평균 차이"].mean()

        a, b, c = st.columns(3)

        a.metric(
            "전략 평균 일치",
            f"{strategy_mean:.4f}",
        )

        b.metric(
            "무작위 평균 일치",
            f"{random_mean:.4f}",
        )

        c.metric(
            "평균 차이",
            f"{diff_mean:+.4f}",
        )

        st.subheader("평균 차이의 95% 부트스트랩 신뢰구간")

        st.write(
            f"[{low_ci:+.4f}, {high_ci:+.4f}]"
        )

        st.caption(
            "신뢰구간은 관측된 평균 차이의 불확실성을 추정합니다. "
            "회차 간 의존성이나 전략 선택 과정의 영향을 완전히 "
            "제거하지는 않으므로 최종적인 성능 증명은 아닙니다."
        )

        st.subheader("30회 이동 평균")

        st.line_chart(
            result.set_index("대상 회차")[
                [
                    "전략 평균 일치",
                    "무작위 평균 일치",
                ]
            ].rolling(
                30,
                min_periods=1,
            ).mean()
        )

        split = max(1, int(len(result) * 0.8))
        holdout = result.iloc[split:]

        if not holdout.empty:
            st.subheader("시간상 마지막 20% 구간")

            st.write(
                "전략 평균: "
                f"{holdout['전략 평균 일치'].mean():.4f}"
            )

            st.write(
                "무작위 평균: "
                f"{holdout['무작위 평균 일치'].mean():.4f}"
            )

            st.write(
                "평균 차이: "
                f"{holdout['평균 차이'].mean():+.4f}"
            )

        if low_ci <= 0 <= high_ci:
            st.info(
                "신뢰구간에 0이 포함됩니다. "
                "이번 실험만으로 전략이 무작위보다 우수하다고 "
                "판단하기 어렵습니다."
            )

        elif low_ci > 0:
            st.info(
                "관측된 평균 차이는 양수입니다. "
                "다른 기간과 검증 방법에서도 재현되는지 확인하세요."
            )

        else:
            st.info(
                "이번 실험에서 전략의 평균 일치 개수가 "
                "무작위 기준보다 낮게 관측됐습니다."
            )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "백테스트 CSV 다운로드",
            result.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="lotto_backtest.csv",
            mime="text/csv",
        )


# ==========================================
# 15. 데이터 품질 점검 및 다운로드
# ==========================================

st.header("6. 데이터 품질 점검")

quality = check_data_quality(df)

quality_df = pd.DataFrame([
    {
        "항목": key,
        "결과": (
            ", ".join(map(str, value))
            if isinstance(value, list)
            else value
        ),
    }
    for key, value in quality.items()
])

st.dataframe(
    quality_df,
    use_container_width=True,
    hide_index=True,
)

if quality["누락 회차 수"] > 0:
    st.warning(
        "회차 누락이 있습니다. 최신 데이터 수집 및 갱신을 실행해 주세요."
    )
else:
    st.success(
        "저장된 회차 번호에 누락이 없습니다."
    )

st.header("7. 전체 데이터 다운로드")

st.download_button(
    "전체 당첨 데이터 CSV 다운로드",
    df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="lotto645.csv",
    mime="text/csv",
)

st.caption(
    "Lotto AI Lab은 통계 분석 및 오락용입니다. "
    "모든 조합의 당첨 확률은 공정한 추첨을 가정할 때 동일합니다.그런데요 제가 심심해서 만들었슴"
)
