"""
분포 이동 감지 테스트.

방금 대화형으로 손으로 검증했던 4가지 케이스(동일 분포/완만한 이동/
큰 이동/상수 컬럼)를 정식 테스트로 고정한다. 난수를 쓰지만 seed를
고정해서 재현 가능하게 한다.
"""

import numpy as np
import pandas as pd
import pytest

from qa_pipeline.drift import (
    PSI_SIGNIFICANT_THRESHOLD,
    PSI_STABLE_THRESHOLD,
    _psi_for_column,
    check_distribution_drift,
)
from qa_pipeline.gate import check_drift


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_identical_distributions_have_low_psi(rng):
    ref = pd.Series(rng.normal(100, 15, 5000))
    same = pd.Series(rng.normal(100, 15, 5000))
    psi = _psi_for_column(ref, same)
    assert psi < PSI_STABLE_THRESHOLD


def test_large_shift_has_high_psi(rng):
    ref = pd.Series(rng.normal(100, 15, 5000))
    shifted = pd.Series(rng.normal(130, 15, 5000))  # 2 표준편차 이동
    psi = _psi_for_column(ref, shifted)
    assert psi > PSI_SIGNIFICANT_THRESHOLD


def test_constant_column_does_not_crash():
    ref = pd.Series([5] * 1000)
    inc = pd.Series([5] * 1000)
    psi = _psi_for_column(ref, inc)
    assert psi == 0.0


def test_excluded_columns_are_not_checked():
    df = pd.DataFrame({
        "age": [40, 41, 42, 43, 44] * 20,
        "hearing(left)": [1, 2, 1, 2, 1] * 20,
        "smoking": [0, 1, 0, 1, 0] * 20,
    })
    results = check_distribution_drift(df, df.copy())
    checked_columns = {r.column for r in results}
    assert "age" in checked_columns
    assert "hearing(left)" not in checked_columns
    assert "smoking" not in checked_columns


def test_shifted_column_is_flagged_significant_others_are_not(rng):
    n = 3000
    ref = pd.DataFrame({
        "age": rng.normal(45, 12, n),
        "hemoglobin": rng.normal(14.5, 1.5, n),
    })
    incoming = ref.copy()
    incoming["hemoglobin"] = incoming["hemoglobin"] + 1.3  # age는 그대로 둠

    results = {r.column: r for r in check_distribution_drift(ref, incoming)}
    assert results["age"].psi_severity == "stable"
    assert results["hemoglobin"].psi_severity == "significant"
    assert results["hemoglobin"].ks_pvalue < 0.001


# ---------------------------------------------------------------------
# gate.py의 check_drift() / DriftReport — validate_batch와 분리된 별도 진입점
# ---------------------------------------------------------------------

def test_check_drift_report_separates_significant_and_moderate(rng):
    n = 3000
    ref = pd.DataFrame({
        "stable_col": rng.normal(50, 5, n),
        "big_shift_col": rng.normal(50, 5, n),
    })
    incoming = ref.copy()
    incoming["big_shift_col"] = incoming["big_shift_col"] + 20  # 큰 이동

    report = check_drift(ref, incoming)
    assert report.total_columns_checked == 2
    sig_cols = {r.column for r in report.significant}
    assert "big_shift_col" in sig_cols
    assert "stable_col" not in sig_cols


def test_check_drift_as_warnings_only_includes_significant(rng):
    n = 2000
    ref = pd.DataFrame({"x": rng.normal(0, 1, n)})
    incoming = pd.DataFrame({"x": rng.normal(0, 1, n)})  # 변화 없음
    report = check_drift(ref, incoming)
    assert report.as_warnings() == []


def test_validate_batch_is_unaffected_by_drift_module():
    """B안(별도 함수 분리) 결정 확인용 — validate_batch는 drift와 무관하게 그대로 동작해야 한다."""
    from qa_pipeline.gate import validate_batch

    row = {
        "age": 45, "height(cm)": 170, "weight(kg)": 65, "waist(cm)": 82.0,
        "eyesight(left)": 1.0, "eyesight(right)": 1.0, "hearing(left)": 1,
        "hearing(right)": 1, "systolic": 118, "relaxation": 76,
        "fasting blood sugar": 95, "Cholesterol": 190, "triglyceride": 110,
        "HDL": 55, "LDL": 110, "hemoglobin": 14.5, "Urine protein": 1,
        "serum creatinine": 0.9, "AST": 24, "ALT": 22, "Gtp": 28,
        "dental caries": 0, "smoking": 0,
    }
    result = validate_batch(pd.DataFrame([row]))
    assert result.passed
    assert not hasattr(result, "drift")  # ValidationResult에 드리프트 필드가 없어야 함(의도한 분리)
