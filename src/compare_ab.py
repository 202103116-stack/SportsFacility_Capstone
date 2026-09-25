"""
A/B 모델 비교: A = 지역 포함(업종·시설유형·시설구분·시도·시군구·면적·연령), B = 지역 제외(업종·시설유형·시설구분·면적·연령)

R1 랜덤 층화 5-fold 3회 반복, HGB는 중첩 튜닝. (시도만 남긴 중간 모델도 참고로 비교)

사용법:
    python src/compare_ab.py

출력: outputs/reports/ab_comparison.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from modeling import (build_logistic, load_labeled, make_strata, nested_oof_hgb, oof_predictions,
                      sample_hgb_configs, HGBModel)
from train_models import evaluate

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)

FEATURE_SETS = {
    "A 지역 포함(시도+시군구)": ["fcob_nm", "ftype_nm", "faci_gb_nm", "cp_nm", "cpb_nm"],
    "중간: 시도까지만": ["fcob_nm", "ftype_nm", "faci_gb_nm", "cp_nm"],
    "B 지역 제외": ["fcob_nm", "ftype_nm", "faci_gb_nm"],
}


def main():
    df, y, track = load_labeled()
    strata = make_strata(y, track)
    configs = sample_hgb_configs(12)
    rows_all = []
    print(f"{'피처 세트 / 모델':34s} | 공식 PR-AUC [95% CI] (배수)   | 공식 상위10% | 자율 PR-AUC [95% CI] (배수)   | 자율 상위10%")
    for set_name, cats in FEATURE_SETS.items():
        model_oofs = {
            "HGB(중첩 튜닝)": [nested_oof_hgb(df, y, track, configs, seed=s, cats=cats)[0] for s in SEEDS],
            "로지스틱": [oof_predictions(lambda: build_logistic(cat_features=cats), df, y, strata, seed=s) for s in SEEDS],
        }
        for model_name, oofs in model_oofs.items():
            rows = evaluate(y, track, oofs)
            for r in rows:
                r["피처 세트"], r["모델"] = set_name, model_name
            rows_all += rows
            g = {r["그룹"]: r for r in rows}
            o, a = g["공식점검"], g["자율점검"]
            print(f"{set_name + ' / ' + model_name:34s} | {o['PR-AUC']:.3f} [{o['PR-AUC CI하한']:.3f}, {o['PR-AUC CI상한']:.3f}] ({o['PR-AUC 배수']:.1f}배) | "
                  f"{o['Recall@10%'] * 100:4.1f}%       | {a['PR-AUC']:.3f} [{a['PR-AUC CI하한']:.3f}, {a['PR-AUC CI상한']:.3f}] ({a['PR-AUC 배수']:.1f}배) | {a['Recall@10%'] * 100:4.1f}%")
    out = pd.DataFrame(rows_all)
    front = ["피처 세트", "모델", "그룹"]
    out = out[front + [c for c in out.columns if c not in front]]
    path = ROOT / "outputs" / "reports" / "ab_comparison.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
