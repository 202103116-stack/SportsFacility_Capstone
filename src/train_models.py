"""
모델 성능 평가: 튜닝된 HGB(트리) vs 로지스틱 vs 랜덤포레스트, 세 가지 검증 방식

  R1  랜덤 층화 5-fold, 3회 반복 — 예측 대상 시설이 학습 시설과 같은 시군구에 있으므로 실제 적용 조건에 가깝다.
      HGB 하이퍼파라미터는 fold 안에서만 고르는 중첩 검증이라 낙관 편향이 없다.
  R2  시군구 단위 GroupKFold — 처음 보는 지역에서도 되는가(시군구 신호의 일반화)
  R3  (시군구, 시설명) 단위 GroupKFold — 같은 곳·같은 이름의 중복이 폴드에 걸치지 않게 한 결과

사용법:
    python src/train_models.py

출력: outputs/reports/model_metrics.csv, data/processed/oof_hgb_tuned.csv
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import (GROUPS, CategoryRateBaseline, HGBModel, bootstrap_ap_ci, build_forest, build_logistic,
                      group_mask, group_metrics, load_labeled, make_strata, nested_oof_hgb, oof_grouped,
                      oof_predictions, sample_hgb_configs)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
N_BOOT = 100


def evaluate(y, track, oofs):
    rows = []
    for group in GROUPS:
        mask = group_mask(track, group)
        per_seed = [group_metrics(y, o, mask) for o in oofs]
        row = {k: (float(np.mean([m[k] for m in per_seed])) if isinstance(per_seed[0][k], float) else per_seed[0][k])
               for k in per_seed[0]}
        row["PR-AUC 반복 표준편차"] = float(np.std([m["PR-AUC"] for m in per_seed]))
        row["PR-AUC CI하한"], row["PR-AUC CI상한"] = bootstrap_ap_ci(y, oofs[0], mask, n_boot=N_BOOT)
        row["그룹"] = group
        rows.append(row)
    return rows


def show(title, results):
    print(f"\n=== {title} ===")
    print(f"{'모델':22s} | 전체 PR-AUC | 공식 PR-AUC | 자율 PR-AUC [95% CI]      | 자율 상위10% 포착(Lift) | 공식 상위10% 포착")
    for name, rows in results.items():
        r = {x["그룹"]: x for x in rows}
        a = r["자율점검"]
        print(f"{name:22s} | {r['전체']['PR-AUC']:.3f}       | {r['공식점검']['PR-AUC']:.3f}       | "
              f"{a['PR-AUC']:.3f} [{a['PR-AUC CI하한']:.3f}, {a['PR-AUC CI상한']:.3f}] | "
              f"{a['Recall@10%'] * 100:4.1f}% ({a['Lift@10%']:.1f})           | {r['공식점검']['Recall@10%'] * 100:4.1f}%")


def main():
    df, y, track = load_labeled()
    strata = make_strata(y, track)
    region = df["cp_nm"].fillna("미상") + "|" + df["cpb_nm"].fillna("미상")
    region_name = region + "|" + df["faci_nm"].fillna("").str.replace(r"\s+", "", regex=True)
    configs = sample_hgb_configs(12)
    print(f"라벨 있는 정상운영 {len(df):,}건, 위험군 {y.sum():,}건 | 시군구 {region.nunique()}개, (시군구,시설명) 그룹 {region_name.nunique():,}개")

    all_rows, results = [], {}

    def add(regime, name, oofs):
        rows = evaluate(y, track, oofs)
        for r in rows:
            r["검증"], r["모델"] = regime, name
        all_rows.extend(rows)
        results.setdefault(regime, {})[name] = rows

    # R1: 랜덤 층화, HGB는 중첩 튜닝
    hgb_oofs, chosen = [], []
    for s in SEEDS:
        oof, ch = nested_oof_hgb(df, y, track, configs, seed=s)
        hgb_oofs.append(oof)
        chosen += ch
    counts = Counter(chosen)
    best_idx = max(counts, key=lambda k: (counts[k], -k))
    best_cfg = configs[best_idx]
    print(f"\n[중첩 튜닝] 15개 fold에서 선택된 설정 횟수: {dict(counts)}")
    print(f"가장 많이 선택된 설정(#{best_idx}): { {k: v for k, v in best_cfg.items() if k not in ('max_iter', 'early_stopping', 'n_iter_no_change', 'validation_fraction')} }")

    models = {
        "업종(ftype) 위험률": lambda: CategoryRateBaseline("ftype_nm"),
        "로지스틱(balanced)": lambda: build_logistic(),
        "랜덤포레스트": build_forest,
        "HGB(튜닝)": lambda: HGBModel(**best_cfg),
    }
    add("R1 랜덤 5-fold (같은 지역)", "HGB(튜닝, 중첩검증)", hgb_oofs)
    for name in ("업종(ftype) 위험률", "로지스틱(balanced)"):
        add("R1 랜덤 5-fold (같은 지역)", name, [oof_predictions(models[name], df, y, strata, seed=s) for s in SEEDS])
    add("R1 랜덤 5-fold (같은 지역)", "랜덤포레스트", [oof_predictions(models["랜덤포레스트"], df, y, strata, seed=SEEDS[0])])

    # R2, R3: 그룹 단위 검증 (설정은 R1에서 가장 많이 선택된 것으로 고정)
    for regime, groups in (("R2 시군구 단위 (처음 보는 지역)", region), ("R3 (시군구,시설명) 단위 중복 제거", region_name)):
        codes = pd.factorize(groups)[0]
        for name, factory in models.items():
            seeds = SEEDS if name != "랜덤포레스트" else SEEDS[:1]
            add(regime, name, [oof_grouped(factory, df, y, strata, codes, seed=s) for s in seeds])

    for regime, res in results.items():
        show(regime, res)

    out = pd.DataFrame(all_rows)
    front = ["검증", "모델", "그룹"]
    out = out[front + [c for c in out.columns if c not in front]]
    path = ROOT / "outputs" / "reports" / "model_metrics.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"faci_cd": df["faci_cd"], "inspection_track": df["inspection_track"], "y": y,
                  "score": hgb_oofs[0]}).to_csv(ROOT / "data" / "processed" / "oof_hgb_tuned.csv", index=False,
                                                encoding="utf-8-sig")
    print(f"\n저장: {path}")

    print("\n=== R1 HGB(튜닝, 중첩검증) 상세 (우선순위 효과: 상위 k% 점검 시) ===")
    for r in results["R1 랜덤 5-fold (같은 지역)"]["HGB(튜닝, 중첩검증)"]:
        print(f"  {r['그룹']:5s} 기준선 {r['기준선'] * 100:5.2f}% | " +
              " | ".join(f"상위{k}% 포착 {r[f'Recall@{k}%'] * 100:4.1f}% (Lift {r[f'Lift@{k}%']:.1f})" for k in (5, 10, 20)))


if __name__ == "__main__":
    main()
