"""
건강검진 데이터 스키마 정의.

설계 원칙 (DECISIONS.md 2026-08-09 "스키마 범위" 참고):
    범위는 이 프로젝트가 갖고 있는 학습 데이터의 min/max를 그대로 베끼지 않고,
    "의학적으로 있을 법한 범위"를 기준으로 잡았다. 학습 데이터의 관측 범위에만
    맞추면, 학습 데이터엔 없었지만 실제로는 유효한 새 값(예: 학습 데이터에
    85세가 최고령이었다고 96세 신규 데이터를 무조건 걷어내는 것)까지 잘못
    거부하게 된다.

    반대로 이 범위를 너무 넓게 잡으면 진짜 이상치(AST=1090 같은)를 못 거른다.
    그래서 각 컬럼마다 "정상 범위보다는 넓지만, 물리적으로 불가능하지는
    않은" 선에서 상한/하한을 잡았다. 이 기준 자체가 트레이드오프이므로
    출처를 주석으로 남겨둔다.

주의: 이 스키마는 "형식이 유효한가"만 본다. "수축기 ≥ 이완기"처럼 여러
컬럼을 같이 봐야 하는 규칙은 다음 마일스톤(도메인 규칙)에서 다룬다.
이 파일에서 일부러 다루지 않는다 — 마일스톤 범위를 벗어나는 체크를
섞으면 나중에 "이 파일이 뭘 책임지는지" 설명하기 애매해지기 때문이다.
"""

import pandera.pandas as pa
from pandera.typing import Series


class HealthCheckupSchema(pa.DataFrameModel):
    """국가건강검진 데이터 1행 = 1인의 검진 결과."""

    age: Series[int] = pa.Field(ge=1, le=120, description="만 나이(세)")

    # --- 신체 계측 ---
    height_cm: Series[int] = pa.Field(
        ge=100, le=250, alias="height(cm)", description="신장, 성인 기준 극단값 포함"
    )
    weight_kg: Series[int] = pa.Field(
        ge=20, le=300, alias="weight(kg)", description="체중"
    )
    waist_cm: Series[float] = pa.Field(
        ge=30, le=200, alias="waist(cm)", description="허리둘레"
    )

    # --- 시력: 0.1~2.5가 일반적인 시력 값, 9.9는 "실명/측정불가"를 뜻하는
    #     별도 코드다(이 데이터셋의 실제 관행). 이걸 몰랐던 팀 프로젝트에서는
    #     이 값이 단순 이상치로 취급되어 팀원마다 다른 임계값(<=2.0 / <=3.0)으로
    #     걸러졌었다 (smoking-viz-project/docs/03_issues_and_troubleshooting.md
    #     Issue 1). 여기서는 9.9를 "유효한 특수값"으로 명시적으로 허용해서
    #     같은 실수를 반복하지 않는다.
    eyesight_left: Series[float] = pa.Field(
        alias="eyesight(left)",
        check_name=True,
        description="0.1~2.5=시력 값, 9.9=실명/측정불가 코드",
    )
    eyesight_right: Series[float] = pa.Field(
        alias="eyesight(right)",
        check_name=True,
        description="0.1~2.5=시력 값, 9.9=실명/측정불가 코드",
    )

    # --- 청력: 1=정상, 2=이상 (이 데이터셋의 코딩 방식) ---
    hearing_left: Series[int] = pa.Field(isin=[1, 2], alias="hearing(left)")
    hearing_right: Series[int] = pa.Field(isin=[1, 2], alias="hearing(right)")

    # --- 혈압 ---
    systolic: Series[int] = pa.Field(ge=60, le=250, description="수축기 혈압(mmHg)")
    relaxation: Series[int] = pa.Field(ge=30, le=150, description="이완기 혈압(mmHg)")

    # --- 혈액 검사 ---
    fasting_blood_sugar: Series[int] = pa.Field(
        ge=20, le=600, alias="fasting blood sugar"
    )
    cholesterol: Series[int] = pa.Field(ge=50, le=500, alias="Cholesterol")
    triglyceride: Series[int] = pa.Field(ge=5, le=2000)
    hdl: Series[int] = pa.Field(ge=2, le=200, alias="HDL")
    ldl: Series[int] = pa.Field(
        ge=1, le=500, alias="LDL", description="500 초과는 극히 드문 임상값으로 간주"
    )
    hemoglobin: Series[float] = pa.Field(ge=3, le=25)
    urine_protein: Series[int] = pa.Field(
        isin=[1, 2, 3, 4, 5, 6], alias="Urine protein", description="요단백 등급(1~6)"
    )
    serum_creatinine: Series[float] = pa.Field(ge=0.1, le=15, alias="serum creatinine")

    # --- 간 효소: 상한을 낮게 잡아 극단적 이상치를 실제로 걸러내는지
    #     아래 tests/test_schema.py 에서 실제 학습 데이터로 검증한다. ---
    ast: Series[int] = pa.Field(
        ge=5, le=1000, alias="AST", description="1000 초과는 급성 간부전 등 예외적 상황"
    )
    alt: Series[int] = pa.Field(ge=5, le=1000, alias="ALT")
    gtp: Series[int] = pa.Field(
        ge=2, le=600, alias="Gtp", description="일반 검진 모집단에서 600 초과는 매우 드묾"
    )

    dental_caries: Series[int] = pa.Field(isin=[0, 1], alias="dental caries")
    smoking: Series[int] = pa.Field(isin=[0, 1], description="타겟 변수")

    @pa.check("eyesight(left)", "eyesight(right)")
    def valid_eyesight(cls, s: Series[float]) -> Series[bool]:
        """0.1~2.5(정상 시력 범위) 또는 9.9(실명/측정불가 코드)만 허용."""
        return s.between(0.1, 2.5) | (s == 9.9)

    class Config:
        # strict=False: 스키마에 없는 새 컬럼이 섞여 들어와도 이 레벨에서는
        # 실패시키지 않는다. 대신 gate.py에서 "경고"로 별도 보고한다.
        # (2026-08-09 리뷰 결정: 성별처럼 나중에 정말 추가되면 좋을 컬럼까지
        # strict=True 때문에 통째로 거부되는 걸 막기 위함. DECISIONS.md 참고.)
        strict = False
        coerce = False  # 타입을 몰래 바꾸지 않고, 안 맞으면 그대로 실패시킨다
