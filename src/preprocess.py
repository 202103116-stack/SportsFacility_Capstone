"""
Option A 전처리: 점검 받기 전에 관측 가능한 속성만으로 모델링용 피처 테이블 생성

사용법:
    python src/preprocess.py

입력:  data/raw/facility_safety_20260917.csv  (스냅샷 고정 — 최신 파일 자동 선택 안 함)
출력:  data/processed/facility_safety_features_20260917.csv

정상운영 시설만 포함하며, 라벨 있는 시설(학습/검증)과 라벨 없는 시설(예측 대상)이 함께 들어 있다.
점검 이후에야 생기는 컬럼(점검일·공개일·시설정보수정일 등)은 예측 대상에서 결측이므로 제외한다.
선별 기준과 근거는 docs/WORKFLOW.md §5 참고.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = "20260917"
REFERENCE_DATE = pd.Timestamp("2026-09-17")
RAW_PATH = ROOT / "data" / "raw" / f"facility_safety_{SNAPSHOT}.csv"
OUT_PATH = ROOT / "data" / "processed" / f"facility_safety_features_{SNAPSHOT}.csv"

CAT_FEATURES = ["fcob_nm", "ftype_nm", "cp_nm", "cpb_nm", "faci_gb_nm"]
NUM_FEATURES = ["area_sqm", "log_area", "age_years"]
FEATURES = CAT_FEATURES + NUM_FEATURES

# 피처로 쓰지 않는 컬럼과 이유 (팀원/재현용 기록)
EXCLUDED_COLUMNS = {
    "schk_visit_ymd": "점검 결과와 함께 생기는 값 — 예측 대상은 100% 결측",
    "schk_open_ymd": "점검 결과와 함께 생기는 값 — 예측 대상은 100% 결측",
    "faci_upd_ymd": "점검 등록과 함께 갱신되는 것으로 의심 — 라벨 유무에 따라 분포가 다름",
    "inout_gbn_nm": "코드 의미 불명 + 라벨 유무에 따라 결측률이 크게 다름",
    "faci_mng_type_cd": "코드 의미 불명 + 라벨 유무에 따라 결측률이 크게 다름",
    "atnm_chk_yn": "(시설구분, 업종)으로 100% 결정되는 값 — 정보 중복, 평가용 그룹변수(inspection_track)로만 사용",
}

_DATE_LIKE = r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$"


def clean_area(raw: pd.Series) -> pd.Series:
    num = pd.to_numeric(raw, errors="coerce")
    integer_part = raw.str.split(".").str[0]
    # 면적 칸에 날짜(YYYYMMDD)나 신고번호가 입력된 경우만 오염으로 본다. 골프장·스키장의 수백만 ㎡는 정상값이라 유지.
    corrupted = integer_part.str.match(_DATE_LIKE, na=False) | (num >= 1e9)
    return num.where(~corrupted & (num > 0))


def facility_age_years(base_ymd: pd.Series) -> pd.Series:
    valid = base_ymd.where(base_ymd.str.match(r"^(19|20)\d{6}$", na=False))
    dates = pd.to_datetime(valid, format="%Y%m%d", errors="coerce")
    # 1900-01-01 같은 더미값과 미래 날짜는 결측 처리
    dates = dates.where((dates.dt.year >= 1950) & (dates <= REFERENCE_DATE))
    return (REFERENCE_DATE - dates).dt.days / 365.25


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["faci_stat_nm"] == "정상운영"].copy()

    grade = df["schk_tot_grd_nm"]
    unexpected = set(grade.dropna().unique()) - {"양호", "주의", "사용중지"}
    if unexpected:
        raise ValueError(f"예상하지 못한 등급 값: {unexpected}")
    risk = grade.map({"양호": 0, "주의": 1, "사용중지": 1})

    area = clean_area(df["faci_gfa"])
    out = pd.DataFrame(
        {
            "faci_cd": df["faci_cd"],
            "faci_nm": df["faci_nm"],
            "fcob_nm": df["fcob_nm"],
            "ftype_nm": df["ftype_nm"],
            "cp_nm": df["cp_nm"],
            "cpb_nm": df["cpb_nm"],
            "faci_gb_nm": df["faci_gb_nm"],
            "area_sqm": area,
            "log_area": np.log1p(area),
            "age_years": facility_age_years(df["base_ymd"]),
            "inspection_track": df["atnm_chk_yn"].map({"Y": "자율점검", "N": "공식점검"}),
            "grade_raw": grade,
            "risk_binary": risk,
            "has_label": risk.notna(),
        }
    )
    return out.reset_index(drop=True)


def print_report(feats: pd.DataFrame) -> None:
    lab = feats[feats["has_label"]]
    unl = feats[~feats["has_label"]]
    print(f"정상운영 시설: {len(feats):,}건 (라벨 있음 {len(lab):,} / 라벨 없음=예측 대상 {len(unl):,})")
    print(f"위험군: {int(lab['risk_binary'].sum()):,}건 ({lab['risk_binary'].mean() * 100:.2f}%)")
    print()
    print("트랙별 (라벨 있음 기준):")
    for track, g in lab.groupby("inspection_track"):
        print(f"  {track}: {len(g):,}건, 위험군 {int(g['risk_binary'].sum()):,}건 ({g['risk_binary'].mean() * 100:.2f}%)"
              f" | 예측 대상 {int((unl['inspection_track'] == track).sum()):,}건")
    print()
    print("피처 결측률 (%)          라벨 있음   라벨 없음")
    for c in FEATURES:
        print(f"  {c:20s} {lab[c].isna().mean() * 100:8.1f}  {unl[c].isna().mean() * 100:9.1f}")
    print()
    big = int((feats["area_sqm"] > 100_000).sum())
    print(f"면적 100,000㎡ 초과 유지: {big}건 / 면적 10㎡ 미만(유지, 확인 필요): {int((feats['area_sqm'] < 10).sum())}건")


def main() -> None:
    raw = pd.read_csv(RAW_PATH, dtype=str)
    feats = build_features(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUT_PATH}\n")
    print_report(feats)


if __name__ == "__main__":
    main()
