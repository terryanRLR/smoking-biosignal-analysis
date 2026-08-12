"""
모델 학습 파이프라인 테스트.

실제 CSV(3만 8천여 건)로 매번 테스트를 돌리면 느리고, pytest 스위트는
원본 데이터 없이도 돌아가야 한다(이 저장소엔 CSV가 없음). 그래서 작은
합성 데이터에 일부러 신호를 심어서, "파이프라인이 안 죽는다"뿐 아니라
"진짜로 그 신호를 학습하는가"까지 확인한다.
"""

import numpy as np
import pandas as pd
import pytest

from qa_pipeline.model import predict, train_model


@pytest.fixture
def synthetic_checkup_data():
    rng = np.random.default_rng(0)
    n = 800
    smoking = rng.choice([0, 1], n, p=[0.633, 0.367])
    return pd.DataFrame({
        "age": rng.integers(20, 80, n),
        "height(cm)": rng.integers(150, 190, n),
        "weight(kg)": rng.integers(45, 100, n),
        "waist(cm)": rng.uniform(60, 100, n),
        "eyesight(left)": rng.choice([0.8, 1.0, 1.2, 9.9], n),
        "eyesight(right)": rng.choice([0.8, 1.0, 1.2, 9.9], n),
        "hearing(left)": rng.choice([1, 2], n),
        "hearing(right)": rng.choice([1, 2], n),
        "systolic": rng.integers(100, 150, n),
        "relaxation": rng.integers(60, 90, n),
        "fasting blood sugar": rng.integers(70, 130, n),
        "Cholesterol": rng.integers(150, 250, n),
        "triglyceride": rng.integers(50, 200, n),
        "HDL": rng.integers(35, 80, n),
        "LDL": rng.integers(70, 180, n),
        # 흡연자일수록 헤모글로빈·Gtp가 높게 — 의도적으로 신호를 심음
        "hemoglobin": rng.normal(14, 1.2, n) + smoking * 1.0,
        "Urine protein": rng.choice([1, 2, 3], n),
        "serum creatinine": rng.uniform(0.5, 1.3, n),
        "AST": rng.integers(15, 40, n),
        "ALT": rng.integers(15, 40, n),
        "Gtp": rng.integers(15, 60, n) + smoking * 15,
        "dental caries": rng.choice([0, 1], n),
        "smoking": smoking,
    })


def test_train_model_runs_end_to_end_without_error(synthetic_checkup_data):
    model, metrics = train_model(synthetic_checkup_data)
    assert model is not None
    assert 0 <= metrics.accuracy <= 1
    assert metrics.n_test == round(len(synthetic_checkup_data) * 0.2)


def test_model_learns_the_injected_signal(synthetic_checkup_data):
    """무작위 추측(ROC-AUC 0.5)보다 뚜렷하게 나아야 한다 — 데이터에 심어둔
    hemoglobin/Gtp 신호를 실제로 학습했다는 증거."""
    _, metrics = train_model(synthetic_checkup_data)
    assert metrics.roc_auc > 0.65, (
        f"ROC-AUC {metrics.roc_auc:.3f} — 심어둔 신호를 제대로 학습하지 "
        "못하고 있을 가능성. 파이프라인 어딘가 피처가 안 들어가고 있는지 확인할 것."
    )


def test_predict_returns_binary_labels(synthetic_checkup_data):
    model, _ = train_model(synthetic_checkup_data)
    preds = predict(model, synthetic_checkup_data.drop(columns=["smoking"]))
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(synthetic_checkup_data)


def test_v1_and_v2_get_identical_test_split_when_same_random_state(synthetic_checkup_data):
    """③버전 비교(McNemar 검정)가 성립하려면 v1·v2가 '같은 샘플'에 대한
    예측을 내야 한다. feature_columns가 달라도 random_state·입력 데이터가
    같으면 train_test_split의 행 분할 자체는 동일해야 한다 — 이 가정이
    깨지면 두 모델의 예측을 짝지을 수 없어 버전 비교 감사기 전체가
    성립하지 않는다."""
    from qa_pipeline.features import FEATURE_COLUMNS_V1, FEATURE_COLUMNS_V2

    _, m1 = train_model(synthetic_checkup_data, feature_columns=FEATURE_COLUMNS_V1, random_state=42)
    _, m2 = train_model(synthetic_checkup_data, feature_columns=FEATURE_COLUMNS_V2, random_state=42)

    assert list(m1.test_index) == list(m2.test_index)
    assert list(m1.y_test) == list(m2.y_test)
