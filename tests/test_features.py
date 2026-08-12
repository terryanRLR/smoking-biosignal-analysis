"""피처 엔지니어링 테스트 — BMI/WHtR 계산과 시력 실명코드 분리 로직 검증."""

import numpy as np
import pandas as pd

from qa_pipeline.features import FEATURE_COLUMNS_V1, FEATURE_COLUMNS_V2, engineer_features


def test_bmi_is_computed_correctly():
    df = pd.DataFrame([{
        "height(cm)": 170, "weight(kg)": 68, "waist(cm)": 82.0,
        "eyesight(left)": 1.0, "eyesight(right)": 1.0,
    }])
    out = engineer_features(df)
    expected = 68 / (1.70 ** 2)
    assert abs(out.loc[0, "BMI"] - expected) < 1e-6


def test_whtr_is_computed_correctly():
    df = pd.DataFrame([{
        "height(cm)": 170, "weight(kg)": 68, "waist(cm)": 85.0,
        "eyesight(left)": 1.0, "eyesight(right)": 1.0,
    }])
    out = engineer_features(df)
    expected = 85.0 / 170
    assert abs(out.loc[0, "WHtR"] - expected) < 1e-6


def test_blind_sentinel_is_split_into_flag_and_nan():
    df = pd.DataFrame([
        {"height(cm)": 170, "weight(kg)": 68, "waist(cm)": 82.0, "eyesight(left)": 1.2, "eyesight(right)": 9.9},
        {"height(cm)": 160, "weight(kg)": 55, "waist(cm)": 70.0, "eyesight(left)": 9.9, "eyesight(right)": 0.8},
    ])
    out = engineer_features(df)

    assert out.loc[0, "eyesight_right_blind"] == 1
    assert np.isnan(out.loc[0, "eyesight(right)"])
    assert out.loc[0, "eyesight_left_blind"] == 0
    assert out.loc[0, "eyesight(left)"] == 1.2

    assert out.loc[1, "eyesight_left_blind"] == 1
    assert np.isnan(out.loc[1, "eyesight(left)"])


def test_normal_eyesight_values_are_untouched():
    df = pd.DataFrame([{
        "height(cm)": 170, "weight(kg)": 65, "waist(cm)": 80.0,
        "eyesight(left)": 1.0, "eyesight(right)": 1.5,
    }])
    out = engineer_features(df)
    assert out.loc[0, "eyesight_left_blind"] == 0
    assert out.loc[0, "eyesight_right_blind"] == 0
    assert out.loc[0, "eyesight(left)"] == 1.0
    assert out.loc[0, "eyesight(right)"] == 1.5


def test_original_dataframe_is_not_mutated():
    df = pd.DataFrame([{
        "height(cm)": 170, "weight(kg)": 68, "waist(cm)": 82.0,
        "eyesight(left)": 9.9, "eyesight(right)": 1.0,
    }])
    engineer_features(df)
    assert "BMI" not in df.columns  # 원본은 그대로여야 함
    assert df.loc[0, "eyesight(left)"] == 9.9  # 원본 값이 NaN으로 바뀌면 안 됨


def test_v1_and_v2_feature_columns_differ_by_exactly_whtr():
    assert set(FEATURE_COLUMNS_V2) - set(FEATURE_COLUMNS_V1) == {"WHtR"}
    assert set(FEATURE_COLUMNS_V1) - set(FEATURE_COLUMNS_V2) == set()


def test_feature_columns_all_present_after_engineering():
    row = {
        "age": 45, "height(cm)": 170, "weight(kg)": 65, "waist(cm)": 82.0,
        "eyesight(left)": 1.0, "eyesight(right)": 1.0, "hearing(left)": 1,
        "hearing(right)": 1, "systolic": 118, "relaxation": 76,
        "fasting blood sugar": 95, "Cholesterol": 190, "triglyceride": 110,
        "HDL": 55, "LDL": 110, "hemoglobin": 14.5, "Urine protein": 1,
        "serum creatinine": 0.9, "AST": 24, "ALT": 22, "Gtp": 28,
        "dental caries": 0, "smoking": 0,
    }
    out = engineer_features(pd.DataFrame([row]))
    for cols in (FEATURE_COLUMNS_V1, FEATURE_COLUMNS_V2):
        missing = [c for c in cols if c not in out.columns]
        assert missing == [], f"엔지니어링 후에도 없는 컬럼: {missing}"
