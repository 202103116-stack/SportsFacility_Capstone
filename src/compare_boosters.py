"""
부스팅 계열 비교: LightGBM, CatBoost vs HGB(scikit-learn). 피처는 A(지역 포함), R1 랜덤 층화 5-fold 3회 반복, 중첩 튜닝.

HGB는 LightGBM과 같은 방식(히스토그램 기반, 리프 중심 성장)이라 결과가 비슷할 것으로 예상하고,
CatBoost는 시군구처럼 값이 많은 범주형을 다루는 방식이 달라 차이가 날 수 있어 시험한다.

사용법:
    python -u src/compare_boosters.py

출력: outputs/reports/booster_comparison.csv
"""

import time
from pathlib import Path

import pandas as pd

from modeling import (catboost_factory, lightgbm_factory, load_labeled, nested_oof_hgb, sample_catboost_configs,
                      sample_lightgbm_configs)
from train_models import evaluate

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)


def main():
    df, y, track = load_labeled()
    runs = {
        "LightGBM(중첩 튜닝)": (lightgbm_factory, sample_lightgbm_configs(12)),
        "CatBoost(중첩 튜닝)": (catboost_factory, sample_catboost_configs(6)),
    }
    rows_all = []
    for name, (factory, configs) in runs.items():
        t = time.time()
        oofs = [nested_oof_hgb(df, y, track, configs, seed=s, factory=factory)[0] for s in SEEDS]
        rows = evaluate(y, track, oofs)
        for r in rows:
            r["모델"] = name
        rows_all += rows
        g = {r["그룹"]: r for r in rows}
        o, a = g["공식점검"], g["자율점검"]
        print(f"[{name}] {time.time() - t:.0f}초 | 전체 {g['전체']['PR-AUC']:.3f} | "
              f"공식 {o['PR-AUC']:.3f} [{o['PR-AUC CI하한']:.3f}, {o['PR-AUC CI상한']:.3f}] 상위10% {o['Recall@10%'] * 100:.1f}% | "
              f"자율 {a['PR-AUC']:.3f} [{a['PR-AUC CI하한']:.3f}, {a['PR-AUC CI상한']:.3f}] (±{a['PR-AUC 반복 표준편차']:.3f}) 상위10% {a['Recall@10%'] * 100:.1f}%",
              flush=True)

    prev = pd.read_csv(ROOT / "outputs" / "reports" / "model_metrics.csv")
    prev = prev[(prev["검증"].str.startswith("R1")) & (prev["모델"].isin(["HGB(튜닝, 중첩검증)", "로지스틱(balanced)"]))]
    out = pd.concat([prev.drop(columns=["검증"]), pd.DataFrame(rows_all)], ignore_index=True)
    front = ["모델", "그룹"]
    out = out[front + [c for c in out.columns if c not in front]]
    path = ROOT / "outputs" / "reports" / "booster_comparison.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {path}", flush=True)


if __name__ == "__main__":
    main()
