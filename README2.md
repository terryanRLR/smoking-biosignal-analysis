# 생체 신호 건강검진 결과통보서 — 흡연 여부 비교 대시보드

내일배움캠프 QA/QC 5기 8조(88하조) 팀 과제에서 쓴 데이터셋을 활용해
**개인이 별도로 제작한 후속 인터랙티브 대시보드**입니다. 팀 산출물 정리
저장소(기획서·발표자료·통합 노트북 재구성)는 별도로 존재하며, 이 저장소는
그것과 독립적입니다.

## 미리보기

`index.html`을 브라우저로 열면 바로 동작합니다. **Chart.js는 파일 안에
직접 포함되어 있어(CDN 아님) 인터넷이 없어도 차트가 그려집니다.** 폰트만
Google Fonts/jsdelivr CDN을 사용하는데, 이건 실패해도 시스템 기본 폰트로
자연스럽게 대체되어 기능에는 영향이 없습니다. 별도 서버나 빌드 과정도
필요 없습니다.

> 이전 버전은 Chart.js를 cdnjs CDN에서 불러왔는데, 해당 CDN이 막혀 있는
> 네트워크(회사망 등)에서 열면 스크립트 전체가 죽어서 **KPI 숫자조차 하나도
> 안 보이는 문제**가 있었습니다. Chart.js 라이브러리 전체(~200KB)를 파일에
> 직접 심어서 이 문제 자체를 없앴습니다. (Playwright로 네트워크 완전 차단
> 상태에서도 정상 렌더링되는 것을 확인했습니다.)

## 무엇이 들어있나

- **`index.html`** — 대시보드 본체. 데이터는 파일 안에 `<script id="dashboardData" type="application/json">`
  태그로 이미 내장되어 있어서, CSV 없이도 그대로 열립니다.
- **`analyze.py`** — `train_dataset.csv`(Kaggle: [Smoker Status Prediction Dataset](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction))를
  읽어서 대시보드에 들어간 모든 수치(상관계수, 그룹 평균, 회귀분석 결과 등)를
  재계산하는 스크립트입니다. 라이선스 문제로 원본 CSV는 이 저장소에
  포함하지 않았습니다.
- **`dashboard_data.json`** — `analyze.py`의 출력물. `index.html`에 내장된
  것과 같은 내용입니다.

## 재계산하려면

```bash
pip install pandas numpy scipy statsmodels
python3 analyze.py /path/to/train_dataset.csv
```

`dashboard_data.json`이 갱신됩니다. 이 내용을 `index.html`의
`<script id="dashboardData">` 태그 안에 그대로 붙여넣으면 대시보드에 반영됩니다.
(빌드 도구 없이 손으로 하는 방식이라 다소 투박하지만, 파일 하나로 배포한다는
제약을 지키기 위한 선택입니다.)

## 이 대시보드에서 새로 한 것 (팀 발표자료 대비)

팀 발표자료는 그룹 평균 막대그래프 위주였습니다. 이 대시보드에서는 원본
CSV를 다시 받아 다음을 추가로 검증했습니다.

- **상호작용 회귀분석 (OLS, `cov_type='HC3'`)**: "고BMI × 흡연" 상호작용
  효과가 통계적으로 유의한지를 지표별로 따로 검정했습니다. 결과는 지표마다
  갈렸습니다 — **ALT만 유의한 양(+)의 상호작용**(p=0.006)이 있었고,
  γ-GTP·AST는 유의하지 않았으며(p=0.48, p=0.20), 허리둘레는 오히려
  **유의한 음(-)의 상호작용**(p<0.001)이 나왔습니다. 즉 발표자료의
  "고BMI+흡연 시너지" 주장은 지표에 따라 근거 강도가 다릅니다. 대시보드
  "가설 3" 섹션에 이 결과를 표와 함께 그대로 노출했습니다.
- **전처리 수치 재검증**: 원본 38,984건 → 전처리 후 38,740건(244건 제거)이
  실제 CSV로 재현되는지 확인했습니다 (재현됨 — 자세한 내용은
  `analyze.py`의 필터 조건 참고).

## 기술 스택

- 데이터 처리: pandas, numpy, scipy(point-biserial correlation), statsmodels(OLS)
- 프론트엔드: 순수 HTML/CSS/JS + Chart.js 4 (빌드 도구 없음)
- 폰트: Pretendard(본문/제목), IBM Plex Mono(데이터·수치)

## 알려진 제약

- 원본 데이터에 성별 변수가 없어 이 대시보드도 동일한 한계를 그대로
  가집니다 (판독 소견 섹션에 명시).
- 상관관계 기반 분석이며 인과관계를 증명하지 않습니다.
- Chart.js/폰트를 CDN에서 불러오므로 완전한 오프라인 환경에서는 스타일이
  깨질 수 있습니다 (레이아웃 자체는 폴백 폰트로도 정상 동작합니다).
