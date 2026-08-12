"""
흡연 여부 예측 모델.

설계 결정 (DECISIONS.md 참고):
    HistGradientBoostingClassifier(scikit-learn)를 썼다. XGBoost/LightGBM
    보다 성능이 크게 떨어지지 않으면서 별도 무거운 의존성을 추가하지
    않아도 되고, 결측값(NaN)을 채우지 않고 그대로 넣을 수 있어
    features.py에서 만든 "실명 시 NaN" 처리와 잘 맞는다.

    평가지표는 accuracy 하나만 보지 않는다. 정확도만 보면 다수 클래스
    (비흡연 63.3%)만 계속 예측해도 63.3%가 나오는 착시가 생길 수 있어서,
    precision/recall/F1/ROC-AUC를 같이 본다.

    EvalMetrics에 y_test/y_pred/test_index를 노출해둔다 — ③버전 비교
    감사기에서 v1·v2를 McNemar 검정으로 비교하려면, 두 모델이 "같은
    테스트 샘플"에 대해 낸 예측을 짝지어야 하기 때문이다. random_state와
    입력 데이터가 같으면 feature_columns가 달라도 train_test_split의
    행 분할 자체는 동일하다 (분할은 행 순서 기준이지 컬럼 내용 기준이
    아님) — 이 가정은 tests/test_model.py에서 확인한다.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from qa_pipeline.features import FEATURE_COLUMNS_V1, TARGET_COLUMN, engineer_features

RANDOM_STATE = 42


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    n_test: int
    y_test: pd.Series = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    test_index: pd.Index = field(repr=False)

    def summary(self) -> str:
        tn, fp, fn, tp = self.confusion.ravel()
        return (
            f"n_test={self.n_test}\n"
            f"Accuracy   {self.accuracy:.3f}\n"
            f"Precision  {self.precision:.3f}\n"
            f"Recall     {self.recall:.3f}\n"
            f"F1         {self.f1:.3f}\n"
            f"ROC-AUC    {self.roc_auc:.3f}\n"
            f"혼동행렬 (행=실제, 열=예측):\n"
            f"              예측 비흡연  예측 흡연\n"
            f"  실제 비흡연   {tn:8d}   {fp:8d}\n"
            f"  실제 흡연     {fn:8d}   {tp:8d}"
        )


def train_model(
    df: pd.DataFrame,
    feature_columns: list | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """검증을 통과한 원본 데이터를 받아 피처 엔지니어링 → 학습/평가까지 수행한다.
    (모델, 평가지표) 튜플을 반환한다.

    feature_columns를 지정하지 않으면 v1 피처 세트를 쓴다. v2와 비교하려면
    두 호출에 같은 df·random_state를 주고 feature_columns만 바꿀 것 —
    그래야 테스트 세트(test_index)가 동일하게 나온다."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS_V1

    engineered = engineer_features(df)
    X = engineered[feature_columns]
    y = engineered[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = HistGradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = EvalMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
        confusion=confusion_matrix(y_test, y_pred),
        n_test=len(y_test),
        y_test=y_test,
        y_pred=y_pred,
        test_index=X_test.index,
    )
    return model, metrics


def predict(model, df: pd.DataFrame, feature_columns: list | None = None) -> np.ndarray:
    """원본(엔지니어링 전) 데이터를 받아 흡연 여부(0/1) 예측을 반환한다."""
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS_V1
    engineered = engineer_features(df)
    return model.predict(engineered[feature_columns])

