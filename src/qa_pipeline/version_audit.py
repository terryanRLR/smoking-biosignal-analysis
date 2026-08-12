"""
③ 모델 버전 비교 감사기.

설계 결정 (DECISIONS.md 참고):
    "v2가 v1보다 정확도가 0.3%p 높다" 같은 숫자만 보고 "더 좋아졌다"고
    판단하면 안 된다 — 그 0.3%p가 진짜 개선인지, 아니면 우연히 이번
    테스트 세트에서만 그렇게 나온 건지 구분이 안 되기 때문이다. 이게
    바로 이 프로젝트가 원래 잡으려던 "통계적 오류"의 정확한 사례다.

    McNemar 검정을 쓴다 — 두 모델을 "같은" 테스트 샘플에 대해 비교할 때
    쓰는 표준 방법. 전체 정확도끼리 t-test 하는 것과 다르게, "둘 중
    하나만 맞춘 샘플"에만 주목해서 그 비대칭이 우연 수준인지 판단한다.
    이산 비교쌍(discordant pairs)이 25개 미만이면 정확검정(exact),
    그 이상이면 연속성 보정 카이제곱 검정을 자동으로 골라 쓴다 —
    이건 통계학에서 일반적으로 권장되는 기준이다.

    p-value만 보고서에 내지 않는다 — 표본이 작아 검정력이 부족한
    경우를 별도로 경고한다(이산 비교쌍이 10개 미만이면 "차이가 있어도
    못 잡아낼 수 있다"고 표시). 이 프로젝트 전체에서 반복된 교훈
    ("숫자 하나만 보고 판단하지 않는다")을 여기서도 지킨다.
"""

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

LOW_POWER_DISCORDANT_THRESHOLD = 10
EXACT_TEST_THRESHOLD = 25


@dataclass
class ComparisonResult:
    n_test: int
    both_correct: int
    only_v1_correct: int
    only_v2_correct: int
    both_wrong: int
    v1_accuracy: float
    v2_accuracy: float
    test_type: str  # "exact" | "chi2_corrected"
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    low_power_warning: bool

    def summary(self) -> str:
        diff = self.v2_accuracy - self.v1_accuracy
        lines = [
            f"n_test={self.n_test}  (v1 acc={self.v1_accuracy:.4f}, v2 acc={self.v2_accuracy:.4f}, "
            f"차이={diff:+.4f})",
            "",
            "짝지은 결과표 (같은 샘플 기준):",
            f"  둘 다 맞춤        {self.both_correct}",
            f"  v1만 맞춤         {self.only_v1_correct}",
            f"  v2만 맞춤         {self.only_v2_correct}",
            f"  둘 다 틀림        {self.both_wrong}",
            "",
            f"McNemar 검정 ({self.test_type}): statistic={self.statistic:.4f}, p={self.p_value:.4f}",
        ]

        if self.low_power_warning:
            lines.append(
                f"⚠️  두 모델이 갈린 샘플이 {self.only_v1_correct + self.only_v2_correct}개뿐이라 "
                "검정력이 낮습니다 — 실제로 차이가 있어도 이 결과가 '유의하지 않음'으로 "
                "나올 수 있습니다. '차이가 없다'가 아니라 '이 데이터로는 판단하기 이르다'로 읽으세요."
            )

        if self.significant:
            winner = "v2" if self.only_v2_correct > self.only_v1_correct else "v1"
            lines.append(f"\n✅ 결론: {winner}가 통계적으로 유의하게 더 낫습니다 (α={self.alpha}).")
        else:
            lines.append(
                f"\n➖ 결론: 통계적으로 유의한 차이가 없습니다 (α={self.alpha}). "
                "정확도 숫자가 다르더라도 우연일 가능성을 배제할 수 없습니다."
            )
        return "\n".join(lines)


def compare_versions(
    y_true, y_pred_v1, y_pred_v2, alpha: float = 0.05
) -> ComparisonResult:
    """같은 테스트 샘플(y_true)에 대한 v1·v2의 예측을 받아 McNemar 검정으로 비교한다.

    y_true, y_pred_v1, y_pred_v2는 반드시 같은 길이·같은 순서(같은 샘플)여야
    한다 — model.py의 EvalMetrics.test_index가 v1·v2 사이에 동일함을
    보장하는 게 이 전제조건이다 (tests/test_model.py에서 확인)."""
    y_true = np.asarray(y_true)
    y_pred_v1 = np.asarray(y_pred_v1)
    y_pred_v2 = np.asarray(y_pred_v2)
    if not (len(y_true) == len(y_pred_v1) == len(y_pred_v2)):
        raise ValueError(
            "y_true/y_pred_v1/y_pred_v2 길이가 다릅니다 — 같은 테스트 세트인지 확인하세요."
        )

    v1_correct = y_pred_v1 == y_true
    v2_correct = y_pred_v2 == y_true

    both_correct = int(np.sum(v1_correct & v2_correct))
    only_v1 = int(np.sum(v1_correct & ~v2_correct))
    only_v2 = int(np.sum(~v1_correct & v2_correct))
    both_wrong = int(np.sum(~v1_correct & ~v2_correct))

    discordant = only_v1 + only_v2
    use_exact = discordant < EXACT_TEST_THRESHOLD

    table = np.array([[both_correct, only_v1], [only_v2, both_wrong]])
    if discordant == 0:
        # mcnemar()는 비교쌍이 0개면 에러를 낸다 — 두 모델이 완전히 동일하게
        # 예측했다는 뜻이므로 p=1.0(차이 없음)으로 직접 처리한다.
        statistic, p_value, test_type = 0.0, 1.0, "no_discordant_pairs"
    else:
        result = mcnemar(table, exact=use_exact, correction=not use_exact)
        statistic, p_value = float(result.statistic), float(result.pvalue)
        test_type = "exact" if use_exact else "chi2_corrected"

    return ComparisonResult(
        n_test=len(y_true),
        both_correct=both_correct,
        only_v1_correct=only_v1,
        only_v2_correct=only_v2,
        both_wrong=both_wrong,
        v1_accuracy=float(np.mean(v1_correct)),
        v2_accuracy=float(np.mean(v2_correct)),
        test_type=test_type,
        statistic=statistic,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        low_power_warning=discordant < LOW_POWER_DISCORDANT_THRESHOLD,
    )
