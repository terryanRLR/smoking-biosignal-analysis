# 이 폴더에 대하여

여기엔 원본 데이터 파일(`train_dataset.csv`)이 들어갑니다. 용량과 라이선스
문제로 저장소에는 포함하지 않았습니다. 아래에서 받아 이 폴더에 두세요.

Kaggle — Smoker Status Prediction Dataset
https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction

`.gitignore`에 이 폴더의 csv 파일은 커밋되지 않도록 이미 설정되어 있습니다.


---

## 배치 방법

노트북은 `data/train_dataset.csv` 를 기대합니다 (원래 Colab `/content/` 경로였던 것을 상대경로로 바꿨습니다).

```bash
cp "원본/train_dataset.csv" data/train_dataset.csv
```

`.gitignore` 의 `*.csv` 로 제외되므로 커밋되지 않습니다.

> 노트북 출력을 어떻게 되살렸는지는 [`docs/05_reconstruction_log.md`](../docs/05_reconstruction_log.md) 참조.
