# 02. 분석 파이프라인

이 문서는 `notebooks/team_final/`에 정리된 실제 코드를 기준으로, 데이터가 어떤
단계를 거쳐 최종 발표자료의 차트로 이어졌는지를 설명합니다. 모든 코드는 원본
통합 노트북(`소스_코드_8팀_..._종합.ipynb`)에서 그대로 가져온 것이며, 여기서는
그 흐름만 정리했습니다.

## 전체 흐름

```
train_dataset.csv
      │
      ▼
[전처리] ── 3개 버전 존재 (아래 참고) ──▶ BMI / 연령대 / high_BMI 파생변수
      │
      ▼
[상관관계 분석] Pearson corr → Point-biserial corr(+p-value) → 히트맵
      │
      ▼
┌─────────────┬─────────────┬──────────────────────┐
│  가설 1       │  가설 2       │  가설 3                │
│ 혈관 건강     │ 간 기능       │ 고BMI × 흡연 상호작용   │
│ (헤모글로빈/  │ (AST/ALT/    │ (OLS 회귀 + 표준화     │
│  혈압/HDL)   │  GTP)        │  위험 점수)            │
└─────────────┴─────────────┴──────────────────────┘
      │
      ▼
[결론] 4개 집단(저BMI×비흡연/흡연, 고BMI×비흡연/흡연) 비교 → 타겟층 선정
```

## 1) 전처리 — 왜 3가지 버전이 존재하는가

원본 노트북에는 시점이 다른 세 개의 전처리 구현이 섞여 있습니다. 셋 다 큰 틀은
"이상치 임계값 필터링 + BMI 파생"으로 같지만, 세부 기준이 다릅니다.

| | 버전 A (탐색 초기) | 버전 B (회귀분석용) | 버전 C (발표 반영본) |
|---|---|---|---|
| 시력 이상치 기준 | `≤ 3.0` | `≤ 3.0` | `≤ 2.0` |
| BMI 구간 | 5구간 (저체중/정상/과체중/비만2단계/고도비만) | 이진 (`high_BMI` = BMI≥25) | 6구간 (저체중/정상/과체중/1·2·3단계 비만) |
| 연령 그룹 | 미사용 | 6구간 (20대~70대+) | 4구간 (청년/중년/장년/노년) |
| 사용 위치 | `notebooks/team_final/01`, `03`, `05`의 "버전 A" | `04`, `05`의 회귀분석 | `01`, `03`, `04`, `05`의 "버전 C" |

발표자료 슬라이드의 BMI 구간 라벨(저체중/정상/과체중/**비만2단계**/고도비만, 5개)은
**버전 A**와 정확히 일치합니다. 슬라이드에 나온 "전처리 후 38,740개(244개 제거)"라는
수치도 버전 A/B의 필터 조건(시력 ≤3.0 기준)과 일관됩니다. 즉:

> **최종 발표 수치의 근거는 대부분 버전 A/B이고, 버전 C(시력 ≤2.0)는 발표에 직접
> 반영되지 않은, 별도로 진행된 재검증 시도로 보입니다.**

이 판단은 `train_dataset.csv` 원본 없이 코드만 비교해서 내린 추정입니다. 정확히
확인하려면 세 버전을 모두 실행해 행 개수를 비교해봐야 합니다 (자세한 내용은
[`03_issues_and_troubleshooting.md`](./03_issues_and_troubleshooting.md#issue-1)).

버전 A의 핵심 필터 코드:

```python
train = df[
    (df['Gtp'] < 500) &
    (df['ALT'] < 500) &
    (df['AST'] < 500) &
    (df['LDL'] < 400) &
    (df['serum creatinine'] < 3.0) &
    (df['eyesight(left)'] <= 3.0) &
    (df['eyesight(right)'] <= 3.0)
].copy()

train['BMI'] = train['weight(kg)'] / (train['height(cm)'] / 100) ** 2
train['BMI_group'] = pd.cut(
    train['BMI'], bins=[0, 18.5, 23, 25, 30, 100],
    labels=['저체중', '정상', '과체중', '비만2단계', '고도비만']
)
```

## 2) 상관관계 분석

세 가지 방식이 순차적으로 시도되었고, 통계적으로 가장 엄밀한 것은 버전 B의
point-biserial 상관계수(연속형 변수 × 이진형 변수용 상관계수)와 p-value입니다.

```python
for col in cont_cols:
    r, p = stats.pointbiserialr(train_clean['smoking'], train_clean[col])
    corr_rows.append([col, r, p])
```

발표자료의 "상관계수의 유의미한 범위(|r|≥0.2 / 0.3 / 0.4)" 기준은 이 결과를
바탕으로 팀이 자체적으로 정한 임계값이며, 통계적으로 표준화된 기준은 아닙니다
(표본 수가 크면 작은 상관계수도 유의확률이 낮게 나오는 경향이 있다는 점을
발표자료에서도 스스로 언급하고 있습니다).

## 3) 가설별 검정 방식

- **가설 1 (혈관 건강)**: 그룹별 평균 비교(막대/박스/바이올린 플롯) 위주. 통계 검정
  (t-test, Mann-Whitney U 등)은 `mannwhitneyu`가 import만 되어 있고 실제 사용된
  흔적은 확인되지 않았습니다 — 시각적 비교에 의존한 결론입니다.
- **가설 2 (간 기능)**: 동일하게 그룹 평균 비교 + 임상 기준치(AST/ALT > 40,
  γ-GTP > 60) 초과 비율 비교.
- **가설 3 (BMI × 흡연 상호작용)**: 유일하게 회귀분석을 사용한 가설입니다.

```python
model_gtp = smf.ols('Gtp ~ smoking * high_BMI', data=train_clean).fit(cov_type='HC3')
model_alt = smf.ols('ALT ~ smoking * high_BMI', data=train_clean).fit(cov_type='HC3')
model_ast = smf.ols('AST ~ smoking * high_BMI', data=train_clean).fit(cov_type='HC3')
```

`smoking * high_BMI`는 흡연과 고BMI 각각의 주효과(main effect)뿐 아니라
**상호작용 효과**(두 조건이 동시에 있을 때 추가로 발생하는 효과)를 함께
추정합니다. `cov_type='HC3'`는 이분산성에 강건한(robust) 표준오차를 사용한
것으로, 이 프로젝트에서 가장 통계적으로 신중하게 설계된 부분입니다. 다만
회귀 결과표(계수, p-value, 신뢰구간)가 발표 슬라이드에는 등장하지 않고
그룹 평균 막대그래프로만 요약되어 있어, **상호작용항이 통계적으로 유의했는지는
발표자료만으로는 알 수 없습니다.**

## 4) 결론 — 표준화 위험 점수

4개 집단(저BMI-비흡연 / 저BMI-흡연 / 고BMI-비흡연 / 고BMI-흡연)에 대해 위험
관련 변수를 z-score로 표준화한 뒤 평균을 내 "위험 점수"를 만들었습니다.

```python
risk_vars_score = ['ALT', 'Gtp', 'waist(cm)', 'HDL_risk']  # HDL은 방향을 뒤집어 위험 방향으로 통일
for col in risk_vars_score:
    risk_df[col + '_z'] = (risk_df[col] - risk_df[col].mean()) / risk_df[col].std()
risk_df['risk_score'] = risk_df[[c + '_z' for c in risk_vars_score]].mean(axis=1)
```

이 점수를 근거로 "고BMI 흡연자"를 최종 타겟 집단으로 선정했습니다
(발표자료 6-2 참고).

## 실행 환경 관련 참고

원본 노트북은 Google Colab 환경을 전제로 작성되어 있습니다
(`/content/train_dataset.csv` 경로 하드코딩, `!apt-get install fonts-nanum` 등).
로컬에서 실행하려면 파일 경로와 한글 폰트 설정을 환경에 맞게 수정해야 합니다.
