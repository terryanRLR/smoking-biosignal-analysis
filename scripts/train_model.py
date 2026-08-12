"""
엔드투엔드 학습 스크립트: 원본 CSV → ①검증 게이트 → 게이트 통과 데이터로 ②모델 학습.

이 스크립트가 이 프로젝트의 핵심 주장을 실제로 증명하는 곳이다 —
"검증 게이트와 모델이 따로 노는 두 개의 코드 조각이 아니라, 실제로
연결된 하나의 파이프라인"이라는 것. 학습 데이터에서 게이트가 걸러낸
86건이 실제로 빠진 채로 학습되는 걸 여기서 확인할 수 있다.

사용법:
    python3 scripts/train_model.py [csv_경로]
    (기본 경로: data/train_dataset.csv)
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qa_pipeline.gate import filter_to_valid_rows, validate_batch  # noqa: E402
from qa_pipeline.model import train_model  # noqa: E402

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "train_dataset.csv"
MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "smoking_classifier_v1.joblib"


def main(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"CSV를 찾을 수 없습니다: {csv_path}")
        print("data/README.md 참고해서 Kaggle에서 받아 두세요.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[1/3] 원본 데이터 로드: {len(df)}건")

    result = validate_batch(df)
    print(f"[2/3] 검증 게이트 실행")
    print(result.summary())
    clean = filter_to_valid_rows(df, result)
    print(f"       → 게이트 통과 {len(clean)}건 (제외 {len(df) - len(clean)}건)만 학습에 사용\n")

    print("[3/3] 모델 학습 중...")
    model, metrics = train_model(clean)
    print("\n=== 평가 결과 (held-out 20%) ===")
    print(metrics.summary())

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\n모델 저장: {MODEL_OUT}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    main(path)
