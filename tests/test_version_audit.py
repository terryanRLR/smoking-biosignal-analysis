"""
버전 비교 감사기 테스트.

방금 대화형으로 검증했던 케이스(동일 예측/뚜렷한 차이/소표본 저검정력)를
정식 테스트로 고정한다.
"""

import numpy as np
import pytest

from qa_pipeline.version_audit import compare_versions


def test_identical_predictions_give_p_value_1():
    y_true = np.array([0, 1, 0, 1, 0] * 20)
    r = compare_versions(y_true, y_true.copy(), y_true.copy())
    assert r.p_value == 1.0
    assert not r.significant
    assert r.test_type == "no_discordant_pairs"


def test_clearly_better_v2_is_detected_as_significant():
    rng = np.random.default_rng(1)
    n = 2000
    y_true = rng.choice([0, 1], n)
    v1_pred = np.where(rng.random(n) < 0.30, 1 - y_true, y_true)   # v1: 30% 오류
    v2_pred = np.where(rng.random(n) < 0.10, 1 - y_true, y_true)   # v2: 10% 오류

    r = compare_versions(y_true, v1_pred, v2_pred)
    assert r.significant
    assert r.only_v2_correct > r.only_v1_correct
    assert r.v2_accuracy > r.v1_accuracy


def test_small_sample_triggers_low_power_warning():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    v1_pred = np.array([0, 1, 0, 1, 0, 1, 0, 0])  # 1개 틀림
    v2_pred = np.array([0, 1, 0, 1, 0, 1, 1, 1])  # 1개 틀림(다른 자리)
    r = compare_versions(y_true, v1_pred, v2_pred)
    assert r.low_power_warning


def test_mismatched_lengths_raise_error():
    with pytest.raises(ValueError):
        compare_versions([0, 1, 0], [0, 1], [0, 1, 0])


def test_v1_can_win_too():
    """비교 방향이 v2 우위로 고정돼있지 않은지 확인 — v1이 더 나은 경우도 잡혀야 한다."""
    rng = np.random.default_rng(2)
    n = 2000
    y_true = rng.choice([0, 1], n)
    v1_pred = np.where(rng.random(n) < 0.10, 1 - y_true, y_true)   # v1이 더 좋음
    v2_pred = np.where(rng.random(n) < 0.30, 1 - y_true, y_true)
    r = compare_versions(y_true, v1_pred, v2_pred)
    assert r.significant
    assert r.only_v1_correct > r.only_v2_correct
    assert r.v1_accuracy > r.v2_accuracy
