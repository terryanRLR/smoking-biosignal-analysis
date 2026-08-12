"""
피처 엔지니어링.

설계 결정 (DECISIONS.md 참고):
    시력(eyesight) 9.9(실명/측정불가 코드)를 원본 수치 그대로 모델에 넣으면
    안 된다 — 척도상 9.9는 "숫자가 크다"는 이유만으로 모델이 "시력이
    매우 좋다"는 반대 신호로 잘못 학습할 위험이 있다. 그래서:
      1) 실명 여부를 별도 이진 피처로 분리하고
      2) 실명인 경우의 원래 시력 값은 결측(NaN)으로 바꾼다.
    이번에 쓰는 모델(HistGradientBoostingClassifier)은 NaN을 네이티브로
    처리할 수 있어서, 결측을 대체값으로 채우지 않고 그대로 둔다.

    BMI는 원본 컬럼엔 없지만, smoking-viz-project 상관관계 분석에서
    이미 핵심 변수로 확인된 만큼 키·체중으로 파생시켜 추가한다.

    WHtR(허리둘레/키 비율)은 v2 후보 피처다 (2026-08-10 결정,
    DECISIONS.md 참고) — waist(cm)이 이미 raw 피처로 들어가 있어서
    WHtR이 정말 새 정보를 주는지 불확실한, 일부러 결과가 뻔하지 않게
    고른 비교 대상이다. v1과 v2가 같은 engineer_features() 출력을
    공유하고 FEATURE_COLUMNS만 다르게 써서, "피처 하나 차이"만 남기고
    나머지 조건은 완전히 동일하게 유지한다 (공정한 비교의 전제조건).
"""

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "age", "height(cm)", "weight(kg)", "waist(cm)", "BMI",
    "eyesight(left)", "eyesight(right)",
    "eyesight_left_blind", "eyesight_right_blind",
    "hearing(left)", "hearing(right)",
    "systolic", "relaxation",
    "fasting blood sugar", "Cholesterol", "triglyceride", "HDL", "LDL",
    "hemoglobin", "Urine protein", "serum creatinine",
    "AST", "ALT", "Gtp", "dental caries",
]

# v1 = 지금까지 써온 피처 세트 (이미 학습·평가·저장 완료)
FEATURE_COLUMNS_V1 = list(BASE_FEATURE_COLUMNS)

# v2 = v1 + WHtR. v1과 다른 건 이 한 줄뿐이어야 한다 — 그래야
# "이 피처 하나가 정말 도움이 됐는가"를 깨끗하게 테스트할 수 있다.
FEATURE_COLUMNS_V2 = list(BASE_FEATURE_COLUMNS) + ["WHtR"]

# 하위 호환 + 기본값 (기존 코드가 FEATURE_COLUMNS를 참조하고 있어서 유지)
FEATURE_COLUMNS = FEATURE_COLUMNS_V1

TARGET_COLUMN = "smoking"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """원본 검진 데이터에 파생 피처를 추가한 새 DataFrame을 반환한다
    (원본은 변경하지 않음). v1/v2 피처를 전부 계산해두고, 실제로 어떤
    컬럼을 쓸지는 model.py에서 FEATURE_COLUMNS_V1/V2로 선택한다."""
    df = df.copy()

    df["BMI"] = df["weight(kg)"] / (df["height(cm)"] / 100) ** 2
    df["WHtR"] = df["waist(cm)"] / df["height(cm)"]

    for side in ("left", "right"):
        col = f"eyesight({side})"
        blind_col = f"eyesight_{side}_blind"
        df[blind_col] = (df[col] == 9.9).astype(int)
        df[col] = df[col].where(df[col] != 9.9, np.nan)

    return df

