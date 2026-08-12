"""
검증 게이트 실행부.

설계 결정 (DECISIONS.md 참고):
    Pandera는 기본적으로 lazy=False(첫 오류에서 바로 예외를 던짐)이다.
    이 게이트는 "QA 리포트"가 목적이므로 lazy=True로 모든 오류를 한 번에
    모아서, 호출하는 쪽이 "무엇이 왜 몇 건 실패했는지"를 한 번에 보게 한다.
    대신 데이터가 아주 크면 모든 실패 케이스를 메모리에 들고 있어야 하니
    느려질 수 있다 — 이건 나중에 "성능" 마일스톤에서 다시 다룬다.

    스키마에 없는 새 컬럼은 하드 실패가 아니라 "경고"로 보고한다
    (2026-08-09 리뷰 결정, DECISIONS.md 참고).

    스키마(단일 컬럼 형식) 체크와 도메인 규칙(여러 컬럼 관계) 체크는
    별도 모듈로 분리했지만, 게이트 사용자 입장에서는 "통과했는가"가
    하나의 질문이어야 하므로 여기서 두 결과를 하나의 리포트로 합친다.

    드리프트 체크는 일부러 validate_batch()에 합치지 않았다 (2026-08-09
    결정, DECISIONS.md 참고) — "기준 분포"라는 별도 입력이 필요해서
    함수 시그니처 자체가 다르고, 책임을 분리하는 게 낫다고 판단.
    check_drift()를 별도로 호출해서 필요할 때 조합해서 쓴다.
"""

from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa

from qa_pipeline.domain_rules import check_domain_rules
from qa_pipeline.drift import check_distribution_drift
from qa_pipeline.schema import HealthCheckupSchema


@dataclass
class ValidationResult:
    passed: bool
    total_rows: int
    failed_row_count: int
    failure_cases: pd.DataFrame = field(repr=False)
    warnings: list = field(default_factory=list)
    domain_violations: list = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        schema_ok = self.failure_cases.empty
        if schema_ok:
            lines.append(f"✅ 스키마 체크 통과 — {self.total_rows}건")
        else:
            by_col = (
                self.failure_cases.groupby("column")["failure_case"]
                .count()
                .sort_values(ascending=False)
            )
            lines.append(
                f"❌ 스키마 체크 FAIL — {self.total_rows}건 중 {self.failed_row_count}건에서 "
                f"{len(self.failure_cases)}개 이슈"
            )
            for col, cnt in by_col.items():
                lines.append(f"  - {col}: {cnt}건")

        if self.domain_violations:
            lines.append(f"❌ 도메인 규칙 FAIL — {len(self.domain_violations)}개 규칙 위반")
            for v in self.domain_violations:
                lines.append(f"  - [{v.rule}] {v.description} ({len(v.row_indices)}건)")
        else:
            lines.append("✅ 도메인 규칙 통과")

        lines.append("")
        lines.append("── 최종: " + ("✅ PASS" if self.passed else "❌ FAIL"))

        if self.warnings:
            lines.append("")
            lines.append("⚠️  경고 (통과는 했지만 확인이 필요함):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def _detect_unknown_columns(df: pd.DataFrame) -> list:
    known = set(HealthCheckupSchema.to_schema().columns.keys())
    unknown = [c for c in df.columns if c not in known]
    if not unknown:
        return []
    return [
        f"스키마에 없는 컬럼 {len(unknown)}개 발견: {', '.join(unknown)} — "
        "거부하지 않고 통과시켰습니다. 의도한 새 데이터라면 스키마에 "
        "반영하는 걸 검토하세요 (예: 성별 컬럼 추가)."
    ]


def validate_batch(df: pd.DataFrame) -> ValidationResult:
    """건강검진 데이터 배치를 스키마 + 도메인 규칙 기준으로 검증한다.

    드리프트는 포함하지 않는다 — 별도로 check_drift()를 쓸 것."""
    warnings = _detect_unknown_columns(df)
    domain_violations = check_domain_rules(df)

    try:
        HealthCheckupSchema.validate(df, lazy=True)
        failure_cases = pd.DataFrame()
        failed_row_count = 0
        schema_passed = True
    except pa.errors.SchemaErrors as e:
        failure_cases = e.failure_cases
        failed_row_count = failure_cases["index"].nunique()
        schema_passed = False

    return ValidationResult(
        passed=schema_passed and not domain_violations,
        total_rows=len(df),
        failed_row_count=failed_row_count,
        failure_cases=failure_cases,
        warnings=warnings,
        domain_violations=domain_violations,
    )


@dataclass
class DriftReport:
    total_columns_checked: int
    significant: list = field(default_factory=list)
    moderate: list = field(default_factory=list)
    all_results: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"분포 이동 체크: {self.total_columns_checked}개 컬럼 대상"]
        if not self.significant and not self.moderate:
            lines.append("✅ 전 컬럼 안정적 (PSI < 0.1)")
            return "\n".join(lines)
        if self.significant:
            lines.append(f"🔴 심각한 이동(PSI≥0.25) {len(self.significant)}개:")
            for r in sorted(self.significant, key=lambda x: -x.psi):
                lines.append(f"  - {r.column}: PSI={r.psi:.4f}")
        if self.moderate:
            lines.append(f"🟡 완만한 이동(0.1≤PSI<0.25) {len(self.moderate)}개:")
            for r in sorted(self.moderate, key=lambda x: -x.psi):
                lines.append(f"  - {r.column}: PSI={r.psi:.4f}")
        return "\n".join(lines)

    def as_warnings(self) -> list:
        """ValidationResult.warnings 리스트에 그대로 이어 붙일 수 있는 형태로 변환.
        완만한(moderate) 이동은 경고에 넣지 않는다 — 노이즈가 많아서
        '심각함'만 사람이 보는 경고로 올린다는 판단(2026-08-09)."""
        return [
            f"분포 이동 감지: {r.column} (PSI={r.psi:.4f}, 심각) — "
            "재학습 필요 여부 검토 권장"
            for r in self.significant
        ]


def check_drift(reference: pd.DataFrame, incoming: pd.DataFrame) -> DriftReport:
    """reference(기준 분포) 대비 incoming(신규 배치)의 분포 이동을 검사한다.
    하드 실패 개념이 없다 — population 수준 신호라 사람이 판단해야 한다."""
    results = check_distribution_drift(reference, incoming)
    return DriftReport(
        total_columns_checked=len(results),
        significant=[r for r in results if r.psi_severity == "significant"],
        moderate=[r for r in results if r.psi_severity == "moderate"],
        all_results=results,
    )


def filter_to_valid_rows(df: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
    """검증 결과에서 실패한 행(스키마 위반 + 도메인 규칙 위반)만 제외하고 반환한다.

    설계 결정 (DECISIONS.md 참고): ②예측 모델 마일스톤에서 "게이트를 통과한
    데이터로만 학습한다"는 이 프로젝트의 원래 취지를 지키기 위해 추가.
    경고(warnings)는 걸러내지 않는다 — 경고는 정의상 통과된 것이기 때문.
    """
    bad_indices = set(result.failure_cases["index"].tolist()) if not result.failure_cases.empty else set()
    for v in result.domain_violations:
        bad_indices.update(v.row_indices)
    if not bad_indices:
        return df.copy()
    return df.drop(index=[i for i in bad_indices if i in df.index]).copy()



