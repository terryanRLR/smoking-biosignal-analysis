"""
실제 데이터로 드리프트 감지기를 검증하는 데모 스크립트.

이 스크립트는 pytest 스위트(tests/)에 포함되지 않는다 — 원본 CSV가
저장소에 없고(용량·라이선스), CI에서 매번 도는 "테스트"라기보다는
"이 도구가 알려진 실제 신호를 잡아내는가"를 사람이 눈으로 확인하는
검증/데모 목적이기 때문이다.

방법: 비흡연자 그룹을 '기준 분포', 흡연자 그룹을 '새로 들어온 배치'라고
가정하고 드리프트를 계산한다. 실제로 흡연 여부에 따라 건강 지표가
다르다는 건 이미 smoking-viz-project에서 상관관계로 확인한 사실이므로,
이 스크립트는 "PSI/KS 기반 드리프트 감지가 그 사실과 같은 결론에
도달하는가"를 보여주는 교차검증이다.

사용법:
    python3 scripts/demo_drift_smoker_vs_nonsmoker.py [csv_경로]
    (기본 경로: data/train_dataset.csv — Kaggle에서 받아서 이 위치에 둘 것)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qa_pipeline.drift import check_distribution_drift  # noqa: E402

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "train_dataset.csv"


def main(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"CSV를 찾을 수 없습니다: {csv_path}")
        print("Kaggle에서 train_dataset.csv를 받아 위 경로에 두거나,")
        print("python3 scripts/demo_drift_smoker_vs_nonsmoker.py <경로> 로 직접 지정하세요.")
        print("(https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction)")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    nonsmoker, smoker = df[df["smoking"] == 0], df[df["smoking"] == 1]
    print(f"기준(비흡연) n={len(nonsmoker)}  vs  신규배치(흡연) n={len(smoker)}\n")

    results = sorted(check_distribution_drift(nonsmoker, smoker), key=lambda r: -r.psi)

    print(f"{'컬럼':16s} {'PSI':>8s}  {'심각도':10s} {'KS p-value':>12s}")
    print("-" * 52)
    for r in results:
        print(f"{r.column:16s} {r.psi:8.4f}  {r.psi_severity:10s} {r.ks_pvalue:12.2e}")

    print(
        "\n참고: 예전 상관관계 분석(smoking-viz-project)의 핵심 변수"
        "(hemoglobin, Gtp, triglyceride, serum creatinine, waist)와"
        " 상위권이 겹치는지 확인해보세요 — 서로 다른 두 통계 방법이"
        " 같은 결론에 도달하면 교차검증이 되는 셈입니다."
    )
    print(
        "\n주의: height(cm)가 최상위권으로 나올 수 있는데, 키가 흡연 때문에"
        " 변할 리는 없으므로 이는 흡연율의 성별 불균형에 의한 혼입(confounding)"
        " 신호일 가능성이 높습니다 — 성별 변수 부재 문제(docs Issue 6)가"
        " 드리프트 수치로 정량적으로 드러난 사례입니다."
    )


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    main(path)
