"""
통계적 분포 이동(distribution drift) 감지.

설계 결정 (DECISIONS.md 참고):
    PSI(Population Stability Index)를 주 지표로, KS-test를 보조 지표로
    쓴다. 이유: 이 프로젝트 데이터는 3만 8천여 건으로 표본이 커서, KS-test의
    p-value만 기준으로 삼으면 거의 항상 "유의함"이 나온다(이미
    smoking-viz-project에서 상관계수 유의성 논의할 때도 나왔던 문제).
    PSI는 업계에서 피처 드리프트 모니터링용으로 널리 쓰이고, 표본 크기에
    상대적으로 덜 민감한 해석 가능한 임계값(0.1 / 0.25)을 제공한다.

    드리프트는 "이 행이 틀렸다"가 아니라 "이 배치 전체의 분포가 기준과
    달라 보인다"는 population 수준의 신호다. 개별 행이 틀렸다고 단정할
    수 없으므로, 스키마·도메인 규칙과 달리 드리프트는 하드 실패가 아니라
    경고로 취급한다 (gate.py에서 결정).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

PSI_STABLE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25

# 드리프트 체크에서 제외하는 컬럼과 이유.
#  - 이진/범주형 컬럼: PSI의 분위수 기반 구간 나누기가 범주형엔 안 맞는다
#    (별도 카테고리별 비율 비교 방식이 필요 — 다음 개선 과제로 남김).
#  - 시력(eyesight): 0.1~2.5 정상값과 9.9(실명 코드)가 섞여 있어 분위수
#    구간을 심하게 왜곡시킨다. 9.9를 분리해서 별도로 다뤄야 하는데
#    이번 마일스톤 범위 밖이라 일단 제외.
EXCLUDED_COLUMNS = {
    "hearing(left)", "hearing(right)", "Urine protein",
    "dental caries", "smoking",
    "eyesight(left)", "eyesight(right)",
}


@dataclass
class DriftResult:
    column: str
    psi: float
    psi_severity: str  # "stable" | "moderate" | "significant"
    ks_statistic: float
    ks_pvalue: float
    reference_n: int
    incoming_n: int


def _psi_severity(psi: float) -> str:
    if psi < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate"
    return "significant"


def _psi_for_column(reference: pd.Series, incoming: pd.Series, bins: int = 10) -> float:
    """reference의 분위수로 구간을 나누고, 그 구간에 incoming이 얼마나
    다르게 분포하는지를 점수화한다. 값이 거의 동일하면 구간이 겹쳐서
    bins보다 적은 구간이 나올 수 있다(np.unique로 중복 경계 제거)."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = reference.quantile(quantiles).to_numpy().copy()
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        # 값이 거의 상수라 구간을 나눌 수 없는 경우 — 드리프트 계산 불가.
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    inc_counts, _ = np.histogram(incoming, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1)
    inc_pct = inc_counts / max(len(incoming), 1)

    eps = 1e-4  # 0으로 나누기/log(0) 방지용 스무딩
    ref_pct = np.clip(ref_pct, eps, None)
    inc_pct = np.clip(inc_pct, eps, None)

    return float(np.sum((inc_pct - ref_pct) * np.log(inc_pct / ref_pct)))


def check_distribution_drift(
    reference: pd.DataFrame, incoming: pd.DataFrame, columns: list | None = None
) -> list:
    """reference(기준 분포) 대비 incoming(신규 배치)의 컬럼별 분포 이동을 계산한다."""
    if columns is None:
        columns = [
            c for c in reference.columns
            if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(reference[c])
        ]

    results = []
    for col in columns:
        if col not in incoming.columns:
            continue
        ref_s = reference[col].dropna()
        inc_s = incoming[col].dropna()
        if ref_s.empty or inc_s.empty:
            continue

        psi = _psi_for_column(ref_s, inc_s)
        ks_stat, ks_p = stats.ks_2samp(ref_s, inc_s)

        results.append(
            DriftResult(
                column=col,
                psi=round(psi, 4),
                psi_severity=_psi_severity(psi),
                ks_statistic=round(float(ks_stat), 4),
                ks_pvalue=float(ks_p),
                reference_n=len(ref_s),
                incoming_n=len(inc_s),
            )
        )
    return results
