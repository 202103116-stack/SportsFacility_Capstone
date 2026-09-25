"""
우선순위 왜곡 완화 방안 비교 (자율 트랙). A 순위가 위험 판정 0건 지역을 우선순위에서 밀어내는 문제(D14)에 대해
세 가지 방안이 상위 10%를 어떻게 바꾸는지 보고, 지역 관행을 걷어낸 순위가 관측 라벨을 얼마나 덜 맞히는지(OOF) 함께 본다.

  방안 1  지역 내 순위   각 시군구 안에서 A 점수 백분위를 매겨 합침 (지역별 비례 할당과 같은 효과)
  방안 2  A·B 혼합      A와 B의 트랙 내 백분위 평균
  방안 3  0건 지역 대체  위험 판정 0건 지역 시설만 B 백분위, 나머지는 A 백분위

사용법 (score_unlabeled.py 실행 후):
    python src/compare_priority_options.py

입력: data/processed/priority_scores_20260917.csv, data/processed/oof_hgb_tuned.csv
출력: outputs/reports/priority_option_comparison.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modeling import load_labeled

ROOT = Path(__file__).resolve().parent.parent
TOP_FRAC = 0.10


def region_of(df):
    return df["cp_nm"].fillna("미상") + " " + df["cpb_nm"].fillna("미상")


def main():
    lab, y, track = load_labeled()
    lab["region"] = region_of(lab)
    g = pd.DataFrame({"r": lab["region"][track == 1], "y": y[track == 1]}).groupby("r")["y"].agg(["sum", "count"])
    zero_regions = set(g[(g["count"] >= 100) & (g["sum"] == 0)].index)
    print(f"위험 판정 0건 지역(라벨 있는 자율 트랙 시설 100곳 이상): {len(zero_regions)}곳")

    # 1) 예측 대상(자율 트랙): 방안별 상위 10%
    p = pd.read_csv(ROOT / "data" / "processed" / "priority_scores_20260917.csv")
    s = p[p["inspection_track"] == "자율점검"].copy()
    s["region"] = region_of(s)
    s["in_zero"] = s["region"].isin(zero_regions)
    s["A"] = s["score_A"].rank(pct=True)
    s["B"] = s["score_B"].rank(pct=True)
    s["방안1 지역 내 순위"] = s.groupby("region")["score_A"].rank(pct=True)
    s["방안2 A·B 혼합"] = (s["A"] + s["B"]) / 2
    s["방안3 0건 지역 대체"] = np.where(s["in_zero"], s["B"], s["A"])

    n, k, n_regions = len(s), int(len(s) * TOP_FRAC), s["region"].nunique()
    top_a = set(s.nlargest(k, "A")["faci_cd"])
    print(f"\n[예측 대상 자율 트랙 {n:,}건, 상위 10% = {k:,}곳] 전체 중 0건 지역 시설 비율 {s['in_zero'].mean() * 100:.1f}%")
    print(f"{'순위':<22}{'A 상위와 겹침':>12}{'0건지역 비율':>12}{'상위에 든 지역 수':>18}")
    rows = []
    for name in ["A", "방안1 지역 내 순위", "방안2 A·B 혼합", "방안3 0건 지역 대체", "B"]:
        top = s.nlargest(k, name)
        overlap = len(top_a & set(top["faci_cd"])) / k * 100
        share_zero = top["in_zero"].mean() * 100
        print(f"{name:<22}{overlap:>11.1f}%{share_zero:>11.1f}%{top['region'].nunique():>13}/{n_regions}곳")
        rows.append({"구분": "예측 대상 상위 10%", "순위": name, "A 상위와 겹침(%)": overlap, "0건 지역 시설 비율(%)": share_zero,
                     "상위에 든 지역 수": top["region"].nunique(), "전체 지역 수": n_regions})

    # 2) 라벨 있는 자율 트랙(OOF): 지역 관행이 섞인 관측 라벨을 얼마나 맞히는가
    o = pd.read_csv(ROOT / "data" / "processed" / "oof_hgb_tuned.csv")
    m = o.merge(lab[["faci_cd", "region"]], on="faci_cd", how="left")
    m = m[m["inspection_track"] == "자율점검"].copy()
    m["지역 내 순위"] = m.groupby("region")["score"].rank(pct=True)
    kk, total = int(len(m) * TOP_FRAC), m["y"].sum()
    print(f"\n[라벨 있는 자율 트랙 OOF {len(m):,}건, 위험 {int(total)}건, 기준선 {m['y'].mean() * 100:.2f}%]")
    for label, col in (("A 점수(지역 관행 포함)", "score"), ("방안1 지역 내 순위(관행 제거)", "지역 내 순위")):
        top = m.nlargest(kk, col)
        ap = average_precision_score(m["y"], m[col])
        print(f"  {label:<24} PR-AUC {ap:.3f} | 상위 10% 포착 {top['y'].sum() / total * 100:.1f}% | 정밀도 {top['y'].mean() * 100:.2f}%")
        rows.append({"구분": "라벨 있는 OOF(자율)", "순위": label, "PR-AUC": ap, "상위 10% 포착(%)": top["y"].sum() / total * 100,
                     "상위 10% 정밀도(%)": top["y"].mean() * 100, "기준선(%)": m["y"].mean() * 100})

    path = ROOT / "outputs" / "reports" / "priority_option_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
