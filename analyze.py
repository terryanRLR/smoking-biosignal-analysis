import sys
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
# 🔧 FIX(정리): 기본 탐색 경로에 data/ 를 추가.
#   이 저장소의 데이터 배치 위치는 data/train_dataset.csv 인데
#   기존 기본값은 저장소 루트만 봐서 인자 없이 실행하면 항상 실패했다.
_CANDIDATES = [os.path.join(HERE, "data", "train_dataset.csv"),
               os.path.join(HERE, "train_dataset.csv"),
               os.path.join(HERE, "원본", "train_dataset.csv")]
RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else next(
    (p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
OUT_PATH = os.path.join(HERE, "dashboard_data.json")

if not os.path.exists(RAW_PATH):
    print(f"train_dataset.csv를 찾을 수 없습니다: {RAW_PATH}")
    print("사용법: python3 analyze.py [csv 경로]")
    print("(Kaggle: https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction)")
    sys.exit(1)

df = pd.read_csv(RAW_PATH)
n_raw = len(df)

# ---------------------------------------------------------------------
# 전처리 (팀 발표자료가 실제로 채택한 "버전 A/B" 기준: 슬라이드의 244개 제거와
# BMI 5구간 라벨이 이 기준과 일치함 — docs/02_analysis_pipeline.md 참고)
# ---------------------------------------------------------------------
clean = df[
    (df["Gtp"] < 500)
    & (df["ALT"] < 500)
    & (df["AST"] < 500)
    & (df["LDL"] < 400)
    & (df["serum creatinine"] < 3.0)
    & (df["eyesight(left)"] <= 3.0)
    & (df["eyesight(right)"] <= 3.0)
].copy()

n_clean = len(clean)
n_removed = n_raw - n_clean

clean["BMI"] = clean["weight(kg)"] / (clean["height(cm)"] / 100) ** 2
bmi_bins = [0, 18.5, 23, 25, 30, 100]
bmi_labels = ["저체중", "정상", "과체중", "비만2단계", "고도비만"]
clean["BMI_group"] = pd.cut(clean["BMI"], bins=bmi_bins, labels=bmi_labels)
clean["high_BMI"] = (clean["BMI"] >= 25).astype(int)

age_bins = [19, 29, 39, 49, 59, 69, 120]
age_labels = ["20대", "30대", "40대", "50대", "60대", "70대+"]
clean["age_group"] = pd.cut(clean["age"], bins=age_bins, labels=age_labels)

clean["htn"] = ((clean["systolic"] >= 140) | (clean["relaxation"] >= 90)).astype(int)

smoker = clean[clean["smoking"] == 1]
nonsmoker = clean[clean["smoking"] == 0]

out = {}

out["meta"] = {
    "n_raw": int(n_raw),
    "n_clean": int(n_clean),
    "n_removed": int(n_removed),
    "smoker_pct": round(float(clean["smoking"].mean()) * 100, 1),
    "nonsmoker_pct": round(100 - float(clean["smoking"].mean()) * 100, 1),
}

# ---------------------------------------------------------------------
# 상관관계 (point-biserial, smoking 대상)
# ---------------------------------------------------------------------
cont_cols = [
    "hemoglobin", "height(cm)", "weight(kg)", "Gtp", "triglyceride",
    "serum creatinine", "waist(cm)", "ALT", "BMI", "relaxation", "systolic",
    "AST", "eyesight(right)", "fasting blood sugar", "eyesight(left)",
    "Cholesterol", "LDL", "age", "HDL",
]
corr_rows = []
for col in cont_cols:
    r, p = stats.pointbiserialr(clean["smoking"], clean[col])
    corr_rows.append({"var": col, "r": round(float(r), 3), "p": float(p)})
corr_rows.sort(key=lambda x: -x["r"])
out["correlation"] = corr_rows

# ---------------------------------------------------------------------
# 가설 1 — 혈관 건강
# ---------------------------------------------------------------------
hemo_by_bmi = (
    clean.groupby(["BMI_group", "smoking"], observed=True)["hemoglobin"]
    .mean().round(2).unstack()
)
out["h1_hemoglobin_by_bmi"] = {
    "labels": bmi_labels,
    "nonsmoker": [float(hemo_by_bmi.loc[g, 0]) for g in bmi_labels],
    "smoker": [float(hemo_by_bmi.loc[g, 1]) for g in bmi_labels],
}

bp_by_age = (
    clean.groupby(["age_group", "smoking"], observed=True)[["systolic", "relaxation"]]
    .mean().round(2)
)
out["h1_bp_by_age"] = {
    "labels": age_labels,
    "systolic_nonsmoker": [float(bp_by_age.loc[(g, 0), "systolic"]) for g in age_labels],
    "systolic_smoker": [float(bp_by_age.loc[(g, 1), "systolic"]) for g in age_labels],
    "relax_nonsmoker": [float(bp_by_age.loc[(g, 0), "relaxation"]) for g in age_labels],
    "relax_smoker": [float(bp_by_age.loc[(g, 1), "relaxation"]) for g in age_labels],
}

htn_by_age = clean.groupby(["age_group", "smoking"], observed=True)["htn"].mean() * 100
smoke_rate_by_age = clean.groupby("age_group", observed=True)["smoking"].mean() * 100
out["h1_htn_by_age"] = {
    "labels": age_labels,
    "nonsmoker": [round(float(htn_by_age.loc[(g, 0)]), 1) for g in age_labels],
    "smoker": [round(float(htn_by_age.loc[(g, 1)]), 1) for g in age_labels],
    "smoking_rate": [round(float(smoke_rate_by_age.loc[g]), 1) for g in age_labels],
}
# 위험비 (40대 예시)
htn_40_ns = float(htn_by_age.loc[("40대", 0)])
htn_40_s = float(htn_by_age.loc[("40대", 1)])
out["h1_risk_ratio_40s"] = round(htn_40_s / htn_40_ns, 2)

hdl_overall = clean.groupby("smoking")["HDL"].mean().round(1)
hdl_by_bmi = clean.groupby(["BMI_group", "smoking"], observed=True)["HDL"].mean().round(1).unstack()
out["h1_hdl"] = {
    "overall_nonsmoker": float(hdl_overall.loc[0]),
    "overall_smoker": float(hdl_overall.loc[1]),
    "by_bmi_labels": bmi_labels,
    "by_bmi_nonsmoker": [float(hdl_by_bmi.loc[g, 0]) for g in bmi_labels],
    "by_bmi_smoker": [float(hdl_by_bmi.loc[g, 1]) for g in bmi_labels],
}

# ---------------------------------------------------------------------
# 가설 2 — 간 기능
# ---------------------------------------------------------------------
liver_overall = clean.groupby("smoking")[["AST", "ALT", "Gtp"]].mean().round(2)
out["h2_liver_overall"] = {
    "nonsmoker": {k: float(liver_overall.loc[0, k]) for k in ["AST", "ALT", "Gtp"]},
    "smoker": {k: float(liver_overall.loc[1, k]) for k in ["AST", "ALT", "Gtp"]},
}

risk_thresholds = {"AST": 40, "ALT": 40, "Gtp": 60}
risk_pct = {}
for k, th in risk_thresholds.items():
    risk_pct[k] = {
        "nonsmoker": round(float((nonsmoker[k] > th).mean()) * 100, 1),
        "smoker": round(float((smoker[k] > th).mean()) * 100, 1),
    }
out["h2_risk_pct"] = risk_pct

# ---------------------------------------------------------------------
# 가설 3 — 고BMI × 흡연 상호작용
# ---------------------------------------------------------------------
def by_bmi_smoking(col):
    t = clean.groupby(["BMI_group", "smoking"], observed=True)[col].mean().round(2).unstack()
    return {
        "labels": bmi_labels,
        "nonsmoker": [float(t.loc[g, 0]) for g in bmi_labels],
        "smoker": [float(t.loc[g, 1]) for g in bmi_labels],
    }

out["h3_gtp_by_bmi"] = by_bmi_smoking("Gtp")
out["h3_alt_by_bmi"] = by_bmi_smoking("ALT")
out["h3_ast_by_bmi"] = by_bmi_smoking("AST")
out["h3_waist_by_bmi"] = by_bmi_smoking("waist(cm)")
out["h3_tg_by_bmi"] = by_bmi_smoking("triglyceride")

interaction_results = {}
for target in ["Gtp", "ALT", "AST", "waist(cm)"]:
    formula = f'Q("{target}") ~ smoking * high_BMI' if "(" in target else f"{target} ~ smoking * high_BMI"
    model = smf.ols(formula, data=clean).fit(cov_type="HC3")
    term = "smoking:high_BMI"
    interaction_results[target] = {
        "coef": round(float(model.params[term]), 3),
        "p": float(model.pvalues[term]),
        "smoking_main": round(float(model.params["smoking"]), 3),
        "high_bmi_main": round(float(model.params["high_BMI"]), 3),
    }
out["h3_interaction_regression"] = interaction_results

# ---------------------------------------------------------------------
# 결론 — 4개 집단 위험 점수
# ---------------------------------------------------------------------
clean["group4"] = np.where(
    clean["high_BMI"] == 1,
    np.where(clean["smoking"] == 1, "고BMI-흡연", "고BMI-비흡연"),
    np.where(clean["smoking"] == 1, "저BMI-흡연", "저BMI-비흡연"),
)
group_order = ["저BMI-비흡연", "저BMI-흡연", "고BMI-비흡연", "고BMI-흡연"]

risk_vars = ["ALT", "Gtp", "waist(cm)"]
risk_z = pd.DataFrame(index=clean.index)
for col in risk_vars:
    risk_z[col] = (clean[col] - clean[col].mean()) / clean[col].std()
risk_z["HDL_risk"] = -(clean["HDL"] - clean["HDL"].mean()) / clean["HDL"].std()
clean["risk_score"] = risk_z.mean(axis=1)

risk_by_group = clean.groupby("group4")["risk_score"].mean()
group_pct = clean["group4"].value_counts(normalize=True) * 100
out["conclusion_risk_score"] = {
    "labels": group_order,
    "risk_score": [round(float(risk_by_group.loc[g]), 3) for g in group_order],
    "pct": [round(float(group_pct.loc[g]), 1) for g in group_order],
}

out["conclusion_group_liver"] = {
    "labels": group_order,
    "ALT": [round(float(clean[clean.group4 == g]["ALT"].mean()), 1) for g in group_order],
    "Gtp": [round(float(clean[clean.group4 == g]["Gtp"].mean()), 1) for g in group_order],
    "waist": [round(float(clean[clean.group4 == g]["waist(cm)"].mean()), 1) for g in group_order],
    "HDL": [round(float(clean[clean.group4 == g]["HDL"].mean()), 1) for g in group_order],
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print(json.dumps(out["meta"], ensure_ascii=False, indent=1))
print("\ncorrelation top5:")
for r in out["correlation"][:5]:
    print(" ", r)
print("\nh3 interaction regression:")
for k, v in out["h3_interaction_regression"].items():
    print(" ", k, v)
print(f"\nsaved -> {OUT_PATH}")
print("\n주의: index.html에는 이 JSON이 이미 <script id=\"dashboardData\"> 태그 안에 그대로 박혀 있습니다.")
print("데이터를 다시 계산해서 대시보드에 반영하려면, 위 파일 내용으로 그 태그 내용을 교체하세요.")
