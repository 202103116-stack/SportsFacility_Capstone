"""
라벨 없는 정상운영 시설(예측 대상)에 위험 점수를 매긴다. A(지역 포함)가 주 모델, B(지역 제외)는 보조.

최종 모델 = 하이퍼파라미터 선택 절차(5-fold 내부 검증)를 라벨 있는 전체 데이터에 적용한 HGB.
이 절차의 성능은 src/train_models.py, src/compare_ab.py의 중첩 검증으로 이미 측정했다.

사용법:
    python src/score_unlabeled.py

출력: data/processed/priority_scores_20260917.csv, outputs/reports/region_score_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from modeling import CAT_FEATURES, HGBModel, sample_hgb_configs, select_config
from preprocess import OUT_PATH

ROOT = Path(__file__).resolve().parent.parent
B_FEATURES = ["fcob_nm", "ftype_nm", "faci_gb_nm"]


def fit_and_score(name, cats, lab, y, track, unl, configs):
    best = select_config(lab, y, track, np.arange(len(lab)), configs, seed=0, inner_splits=5, cats=cats)
    cfg = configs[best]
    shown = {k: v for k, v in cfg.items() if k not in ("max_iter", "early_stopping", "n_iter_no_change", "validation_fraction")}
    print(f"[{name}] 선택된 설정 #{best}: {shown}")
    return HGBModel(cats=cats, **cfg).fit(lab, y).predict_proba(unl)[:, 1]


def main():
    allf = pd.read_csv(OUT_PATH, low_memory=False)
    lab = allf[allf["has_label"]].reset_index(drop=True)
    unl = allf[~allf["has_label"]].reset_index(drop=True)
    y = lab["risk_binary"].astype(int).to_numpy()
    track = (lab["inspection_track"] == "자율점검").astype(int).to_numpy()
    configs = sample_hgb_configs(12)
    print(f"학습 {len(lab):,}건(위험군 {y.sum():,}) → 예측 대상 {len(unl):,}건\n")

    unl["score_A"] = fit_and_score("A 지역 포함", CAT_FEATURES, lab, y, track, unl, configs)
    unl["score_B"] = fit_and_score("B 지역 제외", B_FEATURES, lab, y, track, unl, configs)
    is_self = (unl["inspection_track"] == "자율점검").to_numpy()
    for m in ("A", "B"):
        unl[f"rank_{m}"] = unl[f"score_{m}"].rank(ascending=False, method="first").astype(int)
        unl[f"rank_{m}_track"] = unl.groupby("inspection_track")[f"score_{m}"].rank(ascending=False, method="first").astype(int)

    cols = ["faci_cd", "faci_nm", "fcob_nm", "ftype_nm", "cp_nm", "cpb_nm", "inspection_track", "area_sqm",
            "score_A", "score_B", "rank_A", "rank_B", "rank_A_track", "rank_B_track"]
    out_path = ROOT / "data" / "processed" / "priority_scores_20260917.csv"
    unl[cols].to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== 점수 분포 (평균 예측확률 vs 라벨 있는 시설의 실제 위험 비율) ===")
    for t, name in (("자율점검", "자율"), ("공식점검", "공식")):
        base = y[(lab["inspection_track"] == t).to_numpy()].mean() * 100
        m = unl["inspection_track"] == t
        print(f"  {name}: 예측 대상 {int(m.sum()):,}건 | 평균 점수 A {unl.loc[m, 'score_A'].mean() * 100:.2f}% / B {unl.loc[m, 'score_B'].mean() * 100:.2f}% | 학습 데이터 위험 비율 {base:.2f}%")

    print("\n=== A와 B는 얼마나 다른 순위를 내는가 (자율 트랙 안) ===")
    s = unl[is_self]
    print(f"  순위 상관(Spearman): {spearmanr(s['score_A'], s['score_B'])[0]:.2f}")
    k = int(len(s) * 0.10)
    top_a, top_b = set(s.nsmallest(k, 'rank_A_track')["faci_cd"]), set(s.nsmallest(k, 'rank_B_track')["faci_cd"])
    print(f"  상위 10%({k:,}곳) 겹침: {len(top_a & top_b) / k * 100:.1f}%")

    print("\n=== 한계 확인: 위험 판정 0건 지역(라벨 있는 자율 트랙 시설 100곳 이상)이 A/B 순위에서 어떻게 되는가 ===")
    lab_region = lab["cp_nm"].fillna("미상") + " " + lab["cpb_nm"].fillna("미상")
    g = pd.DataFrame({"r": lab_region[track == 1], "y": y[track == 1]}).groupby("r")["y"].agg(["sum", "count"])
    zero_regions = set(g[(g["count"] >= 100) & (g["sum"] == 0)].index)
    unl_region = unl["cp_nm"].fillna("미상") + " " + unl["cpb_nm"].fillna("미상")
    in_zero = unl_region.isin(zero_regions).to_numpy()
    share_all = in_zero[is_self].mean() * 100
    print(f"  위험 판정 0건 지역 {len(zero_regions)}곳 — 자율 트랙 예측 대상 중 이 지역 시설 비율: {share_all:.1f}%")
    for m in ("A", "B"):
        top = unl[is_self & (unl[f"rank_{m}_track"] <= k).to_numpy()]
        share_top = unl_region[top.index].isin(zero_regions).mean() * 100
        print(f"  {m} 상위 10% 안에서 이 지역 시설 비율: {share_top:.1f}%  (전체 비율 {share_all:.1f}%와 같으면 지역에 무관한 순위)")

    unl["region"] = unl_region
    summary = unl.groupby("region").agg(예측대상수=("faci_cd", "size"), 평균점수_A=("score_A", "mean"), 평균점수_B=("score_B", "mean"),
                                       A상위10퍼센트비율=("rank_A_track", lambda r: (r <= k).mean())).reset_index()
    summary["위험판정0건지역"] = summary["region"].isin(zero_regions)
    summary_path = ROOT / "outputs" / "reports" / "region_score_summary.csv"
    summary.sort_values("예측대상수", ascending=False).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}\n저장: {summary_path}")


if __name__ == "__main__":
    main()
