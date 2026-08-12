"""
검증 게이트 테스트.

두 종류로 나눈다:
1. 일부러 망가뜨린 합성 데이터로 "정말 잡아내는가" 확인 (unit test 성격)
2. 실제 학습 데이터에서 관측했던 극단값(analyze 과정에서 발견한 실제 값)을
   재현해서 "이 스키마가 진짜 데이터의 진짜 문제를 잡아내는가" 확인.
   원본 CSV 전체는 저장소에 포함하지 않으므로(용량·라이선스), 문제가 됐던
   실제 값만 손으로 재현한다. 출처: smoking-biosignal-dashboard/analyze.py
   실행 결과 (컬럼별 min/max).
"""

import pandas as pd
import pytest

from qa_pipeline.gate import validate_batch


def _valid_row(**overrides) -> dict:
    """모든 필드가 정상 범위인 기준 행. 테스트마다 필요한 필드만 덮어쓴다."""
    row = {
        "age": 45,
        "height(cm)": 170,
        "weight(kg)": 65,
        "waist(cm)": 82.0,
        "eyesight(left)": 1.0,
        "eyesight(right)": 1.0,
        "hearing(left)": 1,
        "hearing(right)": 1,
        "systolic": 118,
        "relaxation": 76,
        "fasting blood sugar": 95,
        "Cholesterol": 190,
        "triglyceride": 110,
        "HDL": 55,
        "LDL": 110,
        "hemoglobin": 14.5,
        "Urine protein": 1,
        "serum creatinine": 0.9,
        "AST": 24,
        "ALT": 22,
        "Gtp": 28,
        "dental caries": 0,
        "smoking": 0,
    }
    row.update(overrides)
    return row


def _df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------
# 1. 정상 케이스
# ---------------------------------------------------------------------

def test_valid_batch_passes():
    df = _df(_valid_row(), _valid_row(age=70, smoking=1))
    result = validate_batch(df)
    assert result.passed
    assert result.failed_row_count == 0


# ---------------------------------------------------------------------
# 2. 합성으로 주입한 이상 케이스
# ---------------------------------------------------------------------

def test_impossible_age_is_caught():
    df = _df(_valid_row(age=250))
    result = validate_batch(df)
    assert not result.passed
    assert "age" in result.failure_cases["column"].values


def test_invalid_hearing_code_is_caught():
    # 이 데이터셋의 청력 코드는 1(정상)/2(이상) 두 값만 유효하다.
    df = _df(_valid_row(**{"hearing(left)": 9}))
    result = validate_batch(df)
    assert not result.passed
    assert "hearing(left)" in result.failure_cases["column"].values


def test_unknown_extra_column_produces_warning_not_failure():
    # 2026-08-09 리뷰 결정: strict=True → 완화. 새 컬럼은 더 이상 하드
    # 실패가 아니라 경고로 보고된다 (예: 나중에 성별 컬럼이 추가되는 경우).
    df = _df(_valid_row())
    df["gender"] = ["F"]
    result = validate_batch(df)
    assert result.passed, "새 컬럼 때문에 전체가 실패하면 안 됨 (완화된 정책)"
    assert any("gender" in w for w in result.warnings)


def test_eyesight_sentinel_9_9_is_accepted():
    # 9.9는 "실명/측정불가"를 뜻하는 유효한 특수값 — 실패하면 안 된다.
    df = _df(_valid_row(**{"eyesight(left)": 9.9, "eyesight(right)": 9.9}))
    result = validate_batch(df)
    assert result.passed, (
        "9.9는 유효한 실명 코드인데 거부됐다 — schema.py의 시력 체크 로직을 확인할 것"
    )


def test_eyesight_arbitrary_out_of_range_is_caught():
    # 9.9(코드)도 아니고 정상 시력 범위(0.1~2.5)도 아닌 값은 진짜 이상치다.
    df = _df(_valid_row(**{"eyesight(left)": 5.0}))
    result = validate_batch(df)
    assert not result.passed
    assert "eyesight(left)" in result.failure_cases["column"].values


# ---------------------------------------------------------------------
# 3. 실제 학습 데이터에서 관측된 극단값 재현
#    (출처: smoking-biosignal-dashboard/analyze.py 실행 시 컬럼별 min/max)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "column, real_observed_max",
    [
        ("AST", 1090),          # 스키마 상한 1000
        ("ALT", 2914),          # 스키마 상한 1000
        ("Gtp", 999),           # 스키마 상한 600
        ("LDL", 1860),          # 스키마 상한 500
    ],
)
def test_real_world_extreme_values_are_caught(column, real_observed_max):
    """
    이 값들은 지어낸 게 아니라 실제 train_dataset.csv에 존재했던 값이다.
    원래 팀 프로젝트에서는 임계값 필터로 조용히 제거되고 넘어갔던 값들인데,
    이 스키마로는 '왜' 제거되어야 하는지가 실패 사유로 명시적으로 남는다.
    """
    df = _df(_valid_row(**{column: real_observed_max}))
    result = validate_batch(df)
    assert not result.passed
    assert column in result.failure_cases["column"].values


# ---------------------------------------------------------------------
# 4. 도메인 규칙 (개별 값은 정상이어도 조합이 말이 안 되는 경우)
# ---------------------------------------------------------------------

def test_systolic_not_greater_than_relaxation_is_caught():
    # 79/80처럼 뒤집힌 값 — 둘 다 개별 스키마 범위(60~250 / 30~150) 안에는 든다.
    df = _df(_valid_row(systolic=79, relaxation=80))
    result = validate_batch(df)
    assert not result.passed
    rules = [v.rule for v in result.domain_violations]
    assert "systolic_gt_relaxation" in rules


def test_individually_valid_height_weight_but_implausible_bmi_is_caught():
    # 키 245cm(스키마 범위 100~250 안), 체중 22kg(스키마 범위 20~300 안)
    # — 둘 다 개별로는 통과하지만 조합하면 BMI=3.7로 생존 불가능한 값.
    df = _df(_valid_row(**{"height(cm)": 245, "weight(kg)": 22}))
    result = validate_batch(df)
    assert not result.passed
    rules = [v.rule for v in result.domain_violations]
    assert "derived_bmi_plausible" in rules


def test_normal_bp_and_bmi_combo_has_no_domain_violation():
    df = _df(_valid_row())
    result = validate_batch(df)
    assert result.domain_violations == []


def test_filter_to_valid_rows_drops_both_schema_and_domain_failures():
    from qa_pipeline.gate import filter_to_valid_rows

    rows = [
        _valid_row(),                                      # 0: 정상
        _valid_row(AST=99999),                              # 1: 스키마 위반
        _valid_row(systolic=70, relaxation=110),             # 2: 도메인 규칙 위반
        _valid_row(age=50),                                  # 3: 정상
    ]
    df = _df(*rows)
    result = validate_batch(df)
    clean = filter_to_valid_rows(df, result)
    assert clean.index.tolist() == [0, 3]
