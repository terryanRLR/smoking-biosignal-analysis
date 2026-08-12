"""
도메인 규칙 체크.

schema.py는 "각 값이 형식적으로 유효한가"만 본다(컬럼 하나씩). 여기서는
여러 컬럼을 같이 봐야만 알 수 있는, 의학적으로 반드시 성립해야 하는
관계를 체크한다. 스키마 체크를 전부 통과한 행도 도메인 규칙은 위반할 수
있다 — 예를 들어 수축기=80·이완기=120은 둘 다 개별 범위 안에 들지만
관계가 뒤집혀 있어 현실적으로 불가능하다.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class DomainRuleViolation:
    rule: str
    description: str
    row_indices: list


def check_systolic_greater_than_relaxation(df: pd.DataFrame) -> DomainRuleViolation | None:
    """수축기 혈압은 이완기 혈압보다 항상 커야 한다 (심장이 수축할 때가
    이완할 때보다 압력이 높다는 건 혈압의 정의 그 자체)."""
    bad = df[df["systolic"] <= df["relaxation"]]
    if bad.empty:
        return None
    return DomainRuleViolation(
        rule="systolic_gt_relaxation",
        description="수축기 혈압이 이완기 혈압보다 크지 않음",
        row_indices=bad.index.tolist(),
    )


def check_derived_bmi_plausible(
    df: pd.DataFrame, lo: float = 10.0, hi: float = 80.0
) -> DomainRuleViolation | None:
    """키·체중 각각은 개별 스키마 범위(키 100~250cm, 체중 20~300kg) 안에
    들어도, 조합해서 계산한 BMI가 비현실적일 수 있다 (예: 키 245cm에
    체중 22kg는 둘 다 스키마 통과지만 BMI=3.7로 생존 불가능한 조합).
    이 범위(10~80)는 세계에서 기록된 극단적 BMI 사례까지 포함해 넉넉하게
    잡은 것이라, 이걸 벗어나면 거의 확실히 데이터 오류다."""
    bmi = df["weight(kg)"] / (df["height(cm)"] / 100) ** 2
    bad = df[(bmi < lo) | (bmi > hi)]
    if bad.empty:
        return None
    return DomainRuleViolation(
        rule="derived_bmi_plausible",
        description=f"키/체중으로 계산한 BMI가 현실적 범위({lo}~{hi})를 벗어남",
        row_indices=bad.index.tolist(),
    )


ALL_RULES = [
    check_systolic_greater_than_relaxation,
    check_derived_bmi_plausible,
]


def check_domain_rules(df: pd.DataFrame) -> list:
    """등록된 모든 도메인 규칙을 돌려서 위반 목록을 반환한다."""
    violations = []
    for rule_fn in ALL_RULES:
        result = rule_fn(df)
        if result is not None:
            violations.append(result)
    return violations
