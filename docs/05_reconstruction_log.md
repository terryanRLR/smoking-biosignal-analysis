# 05. 복원 로그 — 노트북 출력을 되살린 기록

이 저장소의 `notebooks/team_final/` 6개는 원본 통합 노트북을 **주제별로 재배치**한 것인데,
분리 시점에 **실행 출력이 전부 초기화**돼 있었습니다.
원본에는 출력 177건·그래프 80개가 살아 있었지만, 분리본에서는 **0건**이었습니다.

즉 **데이터가 없는 사람은 결과를 아예 볼 수 없는 상태**였습니다.
이 문서는 그것을 되살린 기록입니다.

---

## 결과

| 노트북 | 출력 | 그래프 |
|---|---|---|
| [`01_data_load_and_preprocessing`](../notebooks/team_final/01_data_load_and_preprocessing.ipynb) | 15 | 1 |
| [`02_correlation_analysis`](../notebooks/team_final/02_correlation_analysis.ipynb) | 14 | 5 |
| [`03_hypothesis1_vascular_health`](../notebooks/team_final/03_hypothesis1_vascular_health.ipynb) | 48 | **36** |
| [`04_hypothesis2_liver_function`](../notebooks/team_final/04_hypothesis2_liver_function.ipynb) | 19 | 14 |
| [`05_hypothesis3_bmi_smoking_interaction`](../notebooks/team_final/05_hypothesis3_bmi_smoking_interaction.ipynb) | 24 | 20 |
| [`06_conclusion_and_risk_scoring`](../notebooks/team_final/06_conclusion_and_risk_scoring.ipynb) | 7 | 5 |
| **합계** | **127** | **81** |

**실행 에러 0건.** 원본 코드셀 106개 중 **105개가 매칭**됐습니다 (1개는 분리 과정에서 빠진 셀).

실행 환경: Python 3.11 · pandas 3.0 · numpy 2.4 · scipy 1.17 · statsmodels 0.14 · matplotlib 3.11

---

## 🐛 그 과정에서 발견한 것 — 주제별 분리가 실행 순서를 깨뜨렸다

**파일 순서(`01 → 02 → … → 06`)대로 실행하면 에러 47건이 납니다.**

### 원인

`01_data_load_and_preprocessing.ipynb` 는 **버전 A** 와 **버전 B** 두 절로 돼 있습니다.

```
버전 A   df = pd.read_csv(...)          →  train['BMI_group'] = pd.cut(...)   ← 파생 생성
버전 B   train = pd.read_csv(...)       ← 데이터를 다시 읽으면서 BMI_group 이 사라진다
```

그리고 `03_hypothesis1_vascular_health.ipynb` 의 **버전 A** 가 그 `BMI_group` 을 씁니다.

```python
group = train.groupby(['BMI_group', 'smoking'])   # KeyError: 'BMI_group'
```

원본 통합 노트북에서는 이 셀들이 **인접해 있어서** 문제가 없었습니다.
주제별로 나누면서 그 사이에 `01` 의 버전 B 가 끼어든 것입니다.

### 실측

| 실행 방식 | 에러 | 그래프 |
|---|---|---|
| 분리본 파일 순서 (`01→06`) | **47건** | 62 |
| **원본 통합 노트북 셀 순서** | **0건** | **81** |

### 대응

**원본 셀 순서로 실행해 얻은 출력을 분리본의 대응 셀에 이식**했습니다.
그리고 각 노트북 맨 위에 **실행 순서 주의 안내**를 넣었습니다.

> 이 저장소의 노트북은 **읽기 좋게 나눈 것**이지 **순서대로 실행하도록 나눈 것이 아닙니다.**
> 재현이 목적이면 `원본/소스_코드_8팀….ipynb` 를 순서대로 실행하세요.

### 남은 선택지

각 노트북을 진짜 독립 실행 가능하게 만들려면, 절마다 앞에
데이터 로드 + 파생 컬럼 재생성 preamble 을 넣어야 합니다.
**그건 원본에 없던 코드를 넣는 것**이라 이번에는 하지 않았습니다 — 판단이 필요한 사항입니다.

---

## 함께 고친 것

| 항목 | 내용 |
|---|---|
| **데이터 경로** | `/content/train_dataset.csv` (Colab) → `../../data/train_dataset.csv` — 6셀 |
| **폰트 경로** | `/usr/share/fonts/truetype/nanum/NanumGothic.ttf` → `C:/Windows/Fonts/malgun.ttf` |
| **`!pip install`** | 주석 처리 + `requirements.txt` 안내 (셸 매직은 로컬에서 불필요) |
| **`display()`** | IPython 내장 함수 — 실행기가 주입 |

---

## 재현 방법

```bash
pip install -r requirements.txt
# data/train_dataset.csv 배치 (원본/train_dataset.csv 를 복사)

# ① 원본 순서로 (권장 — 에러 0)
jupyter notebook "원본/소스_코드_8팀(생체_신호_기반_흡연_여부_비교_시각화_프로젝트) - 종합.ipynb"

# ② 주제별로 읽기만
jupyter notebook notebooks/team_final/
```

`data/train_dataset.csv` 는 `.gitignore` 로 제외돼 있습니다 — [`data/README.md`](../data/README.md) 참조.

---

## 아직 비어 있는 것

| 항목 | 상태 |
|---|---|
| 미매칭 셀 1개 | 원본 코드셀 106개 중 105개만 분리본에 존재. 어느 셀이 빠졌는지는 추적하지 않음 |
| 노트북 독립 실행 | 위 '남은 선택지' 참조 — preamble 추가 여부는 판단 필요 |
| ~~`dashboard_data.json` 생성 코드~~ | ✅ **확인 완료** — 아래 참조 |

---

## ✅ 대시보드 ETL 체인 검증 (추가 확인)

`smoking-biosignal-dashboard.html` 의 데이터가 어디서 오는지 추적해 **전 구간을 확인**했습니다.

```
data/train_dataset.csv
      │  analyze.py            ← 상관분석 · 가설 1~3 집계 · 교호작용 회귀(statsmodels)
      ▼
dashboard_data.json            17개 최상위 키 · 값 284개
      │  수동 이식
      ▼
smoking-biosignal-dashboard.html  <script id="dashboardData"> 태그에 인라인
```

**재실행 검증**

```bash
python analyze.py data/train_dataset.csv
```

| 항목 | 결과 |
|---|---|
| 전체 값 | 284개 |
| 커밋본과 **완전 일치** | **269개** |
| 차이 | 15개 — **전부 p-value 의 14번째 유효숫자** (예: `2.875240707770994e-94` vs `2.8752407077691053e-94`) |
| HTML 인라인 데이터 ↔ JSON | **일치** (`meta` 포함 17키) |

→ **부동소수점 누적 오차일 뿐 실질 차이는 없습니다. ETL 은 정확히 재현됩니다.**

> ⚠️ 다만 **JSON → HTML 은 수동 이식**입니다.
> `analyze.py` 스스로도 실행 후 이렇게 안내합니다 —
> *"index.html에는 이 JSON이 이미 `<script id="dashboardData">` 태그 안에 그대로 박혀 있습니다.
> 데이터를 다시 계산해서 대시보드에 반영하려면, 위 파일 내용으로 그 태그 내용을 교체하세요."*
>
> 자동화하려면 `analyze.py` 가 HTML 의 해당 태그를 직접 갱신하도록 한 단계 추가하면 됩니다.

> 🐛 **경로 기본값 불일치 — 수정함** — `analyze.py` 의 기본 탐색 경로가 **저장소 루트**(`train_dataset.csv`)
> 하나뿐이라, 이 저장소의 배치 위치(`data/train_dataset.csv`)에서는 인자 없이 실행하면 항상 실패했습니다.
> `data/` → 루트 → `원본/` 순으로 찾도록 고쳤습니다. 이제 `python analyze.py` 만으로 동작합니다.
