"""
③ 버전 비교 감사기 데모: v1(현재 피처) vs v2(v1 + WHtR)를 실제 데이터로
학습해서, WHtR 피처가 정말 도움이 되는지 통계적으로 감사한다.

사용법:
    python3 scripts/compare_versions.py [csv_경로]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qa_pipeline.features import FEATURE_COLUMNS_V1, FEATURE_COLUMNS_V2  # noqa: E402
from qa_pipeline.gate import filter_to_valid_rows, validate_batch  # noqa: E402
from qa_pipeline.model import train_model  # noqa: E402
from qa_pipeline.version_audit import compare_versions  # noqa: E402

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "train_dataset.csv"


def main(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"CSV를 찾을 수 없습니다: {csv_path}")
        print("data/README.md 참고해서 Kaggle에서 받아 두세요.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    result = validate_batch(df)
    clean = filter_to_valid_rows(df, result)
    print(f"게이트 통과 데이터: {len(clean)}건 (원본 {len(df)}건 중 {len(df) - len(clean)}건 제외)\n")

    print("[v1] 학습 중 (기존 피처)...")
    _, metrics_v1 = train_model(clean, feature_columns=FEATURE_COLUMNS_V1, random_state=42)
    print(metrics_v1.summary())
    print()

    print("[v2] 학습 중 (v1 + WHtR)...")
    _, metrics_v2 = train_model(clean, feature_columns=FEATURE_COLUMNS_V2, random_state=42)
    print(metrics_v2.summary())
    print()

    assert list(metrics_v1.test_index) == list(metrics_v2.test_index), (
        "test_index가 다릅니다 — v1/v2가 같은 샘플로 평가되지 않았습니다. "
        "McNemar 비교의 전제조건이 깨진 상태라 진행할 수 없습니다."
    )

    print("=" * 60)
    print("버전 비교 감사 (McNemar 검정)")
    print("=" * 60)
    comparison = compare_versions(metrics_v1.y_test, metrics_v1.y_pred, metrics_v2.y_pred)
    print(comparison.summary())


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    main(path)
