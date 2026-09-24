"""
베이스라인 평가: 업종 위험률 기준선 vs 로지스틱 회귀 (위험군 x 점검트랙 층화 5-fold, 3회 반복)

사용법:
    python src/train_baseline.py

출력: outputs/reports/baseline_metrics.csv, data/processed/oof_logistic_baseline.csv
지표 평균은 반복 3회의 평균, PR-AUC 신뢰구간은 첫 반복 OOF 예측을 부트스트랩한 값이다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from modeling import (GROUPS, CategoryRateBaseline, bootstrap_ap_ci, build_logistic, group_mask,
                      group_metrics, load_labeled, make_strata, oof_predictions)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)

MODELS = {
    "업종(ftype) 위험률 기준선": lambda: CategoryRateBaseline("ftype_nm"),
    "로지스틱(balanced)": lambda: build_logistic(),
    "로지스틱 + 결측 플래그": lambda: build_logistic(add_missing_flags=True),
}


def evaluate(make_model, df, y, track, strata):
    oofs = [oof_predictions(make_model, df, y, strata, seed=s) for s in SEEDS]
    rows = []
    for group in GROUPS:
        mask = group_mask(track, group)
        per_seed = [group_metrics(y, o, mask) for o in oofs]
        row = {k: (np.mean([m[k] for m in per_seed]) if isinstance(per_seed[0][k], float) else per_seed[0][k])
               for k in per_seed[0]}
        row["PR-AUC 반복 표준편차"] = float(np.std([m["PR-AUC"] for m in per_seed]))
        row["PR-AUC CI하한"], row["PR-AUC CI상한"] = bootstrap_ap_ci(y, oofs[0], mask)
        row["그룹"] = group
        rows.append(row)
    return rows, oofs[0]


def main():
    df, y, track = load_labeled()
    strata = make_strata(y, track)
    print(f"라벨 있는 정상운영 시설 {len(df):,}건, 위험군 {y.sum():,}건 ({y.mean() * 100:.2f}%)\n")

    all_rows, oof_logit = [], None
    for name, factory in MODELS.items():
        rows, oof = evaluate(factory, df, y, track, strata)
        for r in rows:
            r["모델"] = name
        all_rows += rows
        if name == "로지스틱(balanced)":
            oof_logit = oof
        print(f"[{name}]")
        for r in rows:
            print(f"  {r['그룹']:5s} 위험군 {r['위험군']:5d}/{r['n']:6d} (기준선 {r['기준선'] * 100:5.2f}%) | "
                  f"PR-AUC {r['PR-AUC']:.3f} [{r['PR-AUC CI하한']:.3f}, {r['PR-AUC CI상한']:.3f}] ({r['PR-AUC 배수']:.1f}배, 반복sd {r['PR-AUC 반복 표준편차']:.3f}) | "
                  f"ROC-AUC {r['ROC-AUC']:.3f} | 상위10% 포착 {r['Recall@10%'] * 100:4.1f}% (Lift {r['Lift@10%']:.1f})")
        print()

    out = pd.DataFrame(all_rows)
    front = ["모델", "그룹"]
    out = out[front + [c for c in out.columns if c not in front]]
    report_path = ROOT / "outputs" / "reports" / "baseline_metrics.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(report_path, index=False, encoding="utf-8-sig")

    oof_path = ROOT / "data" / "processed" / "oof_logistic_baseline.csv"
    pd.DataFrame({"faci_cd": df["faci_cd"], "inspection_track": df["inspection_track"],
                  "y": y, "score": oof_logit}).to_csv(oof_path, index=False, encoding="utf-8-sig")
    print(f"저장: {report_path}\n저장: {oof_path}")


if __name__ == "__main__":
    main()
