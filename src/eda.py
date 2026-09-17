"""
전국체육시설 안전점검 정보(TODZ_API_FACI_SAFETY) EDA

사용법:
    python src/eda.py

data/raw/의 가장 최신 facility_safety_*.csv를 읽어 정제 → 그리드 시각화(outputs/figures/eda/)
→ 요약 통계(docs/eda_findings.md)를 생성한다. notebooks/01_eda.ipynb에서도 이 모듈의
함수를 그대로 import해서 재사용한다(단일 소스 유지).
"""

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures" / "eda"
REPORT_PATH = ROOT / "docs" / "eda_findings.md"
REFERENCE_DATE = pd.Timestamp("2026-09-17")  # 데이터 수집 기준일

PALETTE = sns.color_palette("Set2")
RISK_COLORS = {"정상(양호)": "#4C9F70", "위험군(주의+사용중지)": "#D1495B", "미점검(결측)": "#B0B0B0"}

sns.set_theme(style="whitegrid", font="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def latest_safety_csv() -> Path:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "facility_safety_*.csv")))
    if not files:
        raise FileNotFoundError("data/raw/facility_safety_*.csv 없음 — src/fetch_facility_data.py --target safety 먼저 실행")
    return Path(files[-1])


def load_raw() -> pd.DataFrame:
    return pd.read_csv(latest_safety_csv(), dtype=str)


def _valid_ymd(series: pd.Series) -> pd.Series:
    """YYYYMMDD 8자리 + 1990~2026 범위만 유효 날짜로 파싱, 그 외(더미/오타)는 NaT."""
    mask = series.str.match(r"^(19|20)\d{6}$", na=False)
    cleaned = series.where(mask)
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 타겟: 3클래스 원본 + 이진 위험군 재정의
    df["grade_raw"] = df["schk_tot_grd_nm"]
    df["risk_label"] = np.select(
        [df["schk_tot_grd_nm"] == "양호", df["schk_tot_grd_nm"].isin(["주의", "사용중지"])],
        ["정상(양호)", "위험군(주의+사용중지)"],
        default="미점검(결측)",
    )
    df["risk_binary"] = np.select(
        [df["schk_tot_grd_nm"] == "양호", df["schk_tot_grd_nm"].isin(["주의", "사용중지"])],
        [0, 1],
        default=np.nan,
    )

    # 면적 정제: 숫자 변환 후 0 초과 ~ 100,000㎡(대형 스타디움 포함 여유 상한) 벗어나면 결측 처리
    gfa = pd.to_numeric(df["faci_gfa"], errors="coerce")
    df["area_sqm"] = gfa.where((gfa > 0) & (gfa <= 100_000))

    # 날짜 파생
    df["reg_date"] = _valid_ymd(df["faci_reg_ymd"])
    df["visit_date"] = _valid_ymd(df["schk_visit_ymd"])
    df["facility_age_years"] = (REFERENCE_DATE - df["reg_date"]).dt.days / 365.25
    df["days_since_inspection"] = (REFERENCE_DATE - df["visit_date"]).dt.days

    # 비정상(음수) 값 방어
    df.loc[df["facility_age_years"] < 0, "facility_age_years"] = np.nan
    df.loc[df["days_since_inspection"] < 0, "days_since_inspection"] = np.nan

    return df


def savefig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. 데이터 개요
# ---------------------------------------------------------------------------
def fig_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("1. 데이터 개요 — 결측치 · 운영상태 · 시설구분 · 자율점검대상", fontsize=14, fontweight="bold")

    # 결측률 상위 12개 컬럼
    ax = axes[0, 0]
    na_rate = (df.isna().mean() * 100).sort_values(ascending=False).head(12)
    sns.barplot(x=na_rate.values, y=na_rate.index, ax=ax, color=PALETTE[1])
    ax.set_title("결측률 상위 12개 컬럼")
    ax.set_xlabel("결측률 (%)")
    for i, v in enumerate(na_rate.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    # 운영상태명
    ax = axes[0, 1]
    vc = df["faci_stat_nm"].value_counts()
    sns.barplot(x=vc.values, y=vc.index, ax=ax, color=PALETTE[2])
    ax.set_title("운영상태명 (faci_stat_nm)")
    ax.set_xlabel("시설 수")
    for i, v in enumerate(vc.values):
        ax.text(v + 500, i, f"{v:,} ({v/len(df)*100:.1f}%)", va="center", fontsize=9)

    # 시설구분명
    ax = axes[1, 0]
    vc = df["faci_gb_nm"].value_counts()
    ax.pie(vc.values, labels=[f"{k}\n{v:,}건" for k, v in vc.items()], autopct="%1.1f%%",
           colors=PALETTE, startangle=90)
    ax.set_title("시설구분명 (faci_gb_nm)")

    # 자율점검대상여부
    ax = axes[1, 1]
    vc = df["atnm_chk_yn"].value_counts()
    ax.pie(vc.values, labels=[f"{k}\n{v:,}건" for k, v in vc.items()], autopct="%1.1f%%",
           colors=PALETTE[3:], startangle=90)
    ax.set_title("자율점검대상여부 (atnm_chk_yn)")

    return savefig(fig, "01_overview")


# ---------------------------------------------------------------------------
# 2. 타겟 변수
# ---------------------------------------------------------------------------
def fig_target(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("2. 타겟 변수(schk_tot_grd_nm) 분포와 재정의", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    vc = df["grade_raw"].fillna("결측(미점검)").value_counts()
    order = [v for v in ["양호", "주의", "사용중지", "결측(미점검)"] if v in vc.index]
    sns.barplot(x=vc.loc[order].values, y=order, ax=ax, palette=PALETTE)
    ax.set_xscale("symlog")
    ax.set_title("원본 3-클래스 등급 분포 (symlog 스케일)")
    ax.set_xlabel("건수")
    for i, v in enumerate(vc.loc[order].values):
        ax.text(v, i, f"  {v:,} ({v/len(df)*100:.1f}%)", va="center", fontsize=9)

    ax = axes[0, 1]
    vc2 = df["risk_label"].value_counts()
    colors = [RISK_COLORS[k] for k in vc2.index]
    ax.pie(vc2.values, labels=[f"{k}\n{v:,}건 ({v/len(df)*100:.1f}%)" for k, v in vc2.items()],
           colors=colors, startangle=90)
    ax.set_title("재정의된 이진 타겟 (risk_binary)")

    ax = axes[1, 0]
    ct = pd.crosstab(df["faci_stat_nm"], df["grade_raw"].isna().map({True: "라벨 결측", False: "라벨 있음"}))
    ct = ct.reindex(["정상운영", "폐업", "휴업", "운영폐쇄"])
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#4C9F70", "#D1495B"])
    ax.set_title("운영상태별 라벨 결측 여부")
    ax.set_xlabel("")
    ax.set_ylabel("시설 수")
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    normal = df[df["faci_stat_nm"] == "정상운영"]
    vc3 = normal["risk_label"].value_counts()
    colors3 = [RISK_COLORS[k] for k in vc3.index]
    ax.pie(vc3.values, labels=[f"{k}\n{v:,}건 ({v/len(normal)*100:.1f}%)" for k, v in vc3.items()],
           colors=colors3, startangle=90)
    ax.set_title(f"정상운영 시설만(n={len(normal):,})의 타겟 분포")

    return savefig(fig, "02_target")


# ---------------------------------------------------------------------------
# 3. 범주형 피처 분포
# ---------------------------------------------------------------------------
def fig_categorical(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("3. 범주형 예측 피처 분포", fontsize=14, fontweight="bold")

    specs = [
        ("fcob_nm", "업종명 (상위 10)", 10),
        ("ftype_nm", "시설유형명 (상위 10)", 10),
        ("cp_nm", "시도명 (전체)", 16),
        ("cpb_nm", "시군구명 (상위 10)", 10),
        ("faci_mng_type_cd", "시설운영형태코드", None),
        ("inout_gbn_nm", "실내외구분명(코드 의미 불명)", None),
    ]
    for ax, (col, title, topn) in zip(axes.flat, specs):
        vc = df[col].value_counts()
        if topn:
            vc = vc.head(topn)
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color=PALETTE[0])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("시설 수")

    return savefig(fig, "03_categorical")


# ---------------------------------------------------------------------------
# 4. 타겟 대비 위험군 비율 — 범주형 (핵심 인사이트)
# ---------------------------------------------------------------------------
def _risk_rate_by(df: pd.DataFrame, col: str, min_n: int = 30, topn: int = 15) -> pd.DataFrame:
    sub = df.dropna(subset=["risk_binary"])
    g = sub.groupby(col)["risk_binary"].agg(["mean", "count"])
    g = g[g["count"] >= min_n].sort_values("mean", ascending=False).head(topn)
    return g


def fig_risk_rate(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("4. 범주별 위험군(주의+사용중지) 비율 — 최소 표본 30건 이상", fontsize=14, fontweight="bold")

    specs = [("fcob_nm", "업종별 위험 비율 (상위 15)"),
             ("ftype_nm", "시설유형별 위험 비율 (상위 15)"),
             ("cp_nm", "시도별 위험 비율"),
             ("faci_gb_nm", "시설구분별 위험 비율")]

    for ax, (col, title) in zip(axes.flat, specs):
        g = _risk_rate_by(df, col, min_n=30 if col != "faci_gb_nm" else 5, topn=15)
        sns.barplot(x=(g["mean"] * 100), y=g.index.astype(str), ax=ax, color="#D1495B")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("위험군 비율 (%)")
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(m * 100, i, f"  {m*100:.1f}% (n={n:,})", va="center", fontsize=8)

    overall = df["risk_binary"].mean() * 100
    fig.text(0.5, 0.01, f"전체 평균 위험 비율: {overall:.2f}%  (점선 = 전체 평균)", ha="center", fontsize=10)
    for ax in axes.flat:
        ax.axvline(overall, color="gray", linestyle="--", linewidth=1)

    return savefig(fig, "04_risk_rate_by_category")


# ---------------------------------------------------------------------------
# 5. 수치형 피처 분포
# ---------------------------------------------------------------------------
def fig_numeric(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("5. 수치형(파생) 피처 분포", fontsize=14, fontweight="bold")

    ax = axes[0]
    sns.histplot(df["area_sqm"].dropna(), bins=50, ax=ax, color=PALETTE[0], log_scale=(True, False))
    ax.set_title(f"시설 면적(㎡) 분포 (정제 후, n={df['area_sqm'].notna().sum():,})\n로그축, 100,000㎡ 초과·0 이하 제외")
    ax.set_xlabel("면적(㎡, log)")

    ax = axes[1]
    sns.histplot(df["facility_age_years"].dropna(), bins=40, ax=ax, color=PALETTE[1])
    ax.set_title(f"시설연령(년) 분포 (n={df['facility_age_years'].notna().sum():,})\n= 기준일 - 시설정보등록일")
    ax.set_xlabel("연령(년)")

    ax = axes[2]
    sns.histplot(df["days_since_inspection"].dropna(), bins=40, ax=ax, color=PALETTE[2])
    ax.set_title(f"마지막 점검 후 경과일수 분포 (n={df['days_since_inspection'].notna().sum():,})")
    ax.set_xlabel("경과일수")

    return savefig(fig, "05_numeric_distributions")


# ---------------------------------------------------------------------------
# 6. 타겟 대비 수치형 피처
# ---------------------------------------------------------------------------
def fig_numeric_vs_target(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("6. 위험군 여부에 따른 수치형 피처 비교", fontsize=14, fontweight="bold")

    plot_df = df.dropna(subset=["risk_label"]).copy()
    plot_df = plot_df[plot_df["risk_label"] != "미점검(결측)"]
    order = ["정상(양호)", "위험군(주의+사용중지)"]
    palette = [RISK_COLORS[k] for k in order]

    specs = [("area_sqm", "면적(㎡, log)", True),
             ("facility_age_years", "시설연령(년)", False),
             ("days_since_inspection", "점검 경과일수", False)]

    for ax, (col, title, logscale) in zip(axes, specs):
        sns.boxplot(data=plot_df, x="risk_label", y=col, order=order, hue="risk_label",
                    palette=palette, legend=False, ax=ax, showfliers=False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        if logscale:
            ax.set_yscale("log")

    return savefig(fig, "06_numeric_vs_target")


# ---------------------------------------------------------------------------
# 7. 결측치 패턴
# ---------------------------------------------------------------------------
def fig_missing_pattern(df: pd.DataFrame):
    cols = ["faci_daddr", "schk_tot_grd_nm", "faci_mng_type_cd", "inout_gbn_nm", "faci_zip",
            "faci_gfa", "faci_tel_no", "faci_addr", "faci_road_daddr", "sdwn_ymd",
            "faci_homepage", "th_ymd"]
    fig, ax = plt.subplots(figsize=(11, 6))
    miss = df[cols].isna().astype(int)
    sample = miss.sample(min(3000, len(miss)), random_state=42).sort_values(by=cols)
    sns.heatmap(sample.T, cbar=False, cmap=["#E8E8E8", "#D1495B"], ax=ax)
    ax.set_title("7. 결측치 패턴 (샘플 3,000행, 붉은색=결측)", fontsize=13, fontweight="bold")
    ax.set_xlabel("시설 샘플")
    ax.set_xticks([])
    return savefig(fig, "07_missing_pattern")


# ---------------------------------------------------------------------------
# 8. 상관관계
# ---------------------------------------------------------------------------
def fig_correlation(df: pd.DataFrame):
    num_cols = ["risk_binary", "area_sqm", "facility_age_years", "days_since_inspection"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("8. 수치형 피처 상관관계", fontsize=12, fontweight="bold")
    return savefig(fig, "08_correlation")


# ---------------------------------------------------------------------------
# 요약 리포트
# ---------------------------------------------------------------------------
def write_findings_report(df: pd.DataFrame, fig_paths: dict):
    n = len(df)
    risk_n = df["risk_binary"].notna().sum()
    risk_rate = df["risk_binary"].mean() * 100
    normal_df = df[df["faci_stat_nm"] == "정상운영"]

    top_fcob = _risk_rate_by(df, "fcob_nm").head(5)
    top_region = _risk_rate_by(df, "cp_nm", min_n=30).head(5)

    lines = []
    lines.append("# EDA 결과 요약")
    lines.append("")
    lines.append(f"대상: `data/raw/{latest_safety_csv().name}` (TODZ_API_FACI_SAFETY, {n:,}행). "
                 f"생성 스크립트: `src/eda.py` / 그림: `outputs/figures/eda/`. 기준일: {REFERENCE_DATE.date()}.")
    lines.append("")
    lines.append("## 1. 데이터 개요")
    lines.append(f"- 전체 {n:,}건, 운영상태 정상운영 {(df['faci_stat_nm']=='정상운영').mean()*100:.1f}%, "
                 f"폐업/휴업/운영폐쇄 {(df['faci_stat_nm']!='정상운영').mean()*100:.1f}%")
    lines.append(f"- 시설구분: 신고업 {(df['faci_gb_nm']=='신고업').mean()*100:.1f}%, "
                 f"공공 {(df['faci_gb_nm']=='공공').mean()*100:.1f}%, "
                 f"등록업 {(df['faci_gb_nm']=='등록업').mean()*100:.1f}%")
    lines.append(f"- ![데이터 개요]({fig_paths['overview'].relative_to(ROOT).as_posix()})")
    lines.append("")
    lines.append("## 2. 타겟 변수")
    lines.append(f"- 점검 이력 있는 시설: {risk_n:,}건 / 라벨 결측: {n-risk_n:,}건({(n-risk_n)/n*100:.1f}%)")
    lines.append(f"- **전체 위험군(주의+사용중지) 비율: {risk_rate:.2f}%** — 이진분류 기준 심각한 클래스 불균형")
    lines.append(f"- 정상운영 시설만(n={len(normal_df):,}) 봐도 라벨 결측 "
                 f"{normal_df['grade_raw'].isna().mean()*100:.1f}% — 폐업 시설만의 문제가 아님")
    lines.append(f"- ![타겟 변수]({fig_paths['target'].relative_to(ROOT).as_posix()})")
    lines.append("")
    lines.append("## 3. 범주형 피처 분포")
    lines.append(f"- ![범주형 분포]({fig_paths['categorical'].relative_to(ROOT).as_posix()})")
    lines.append("")
    lines.append("## 4. 범주별 위험 비율 (핵심 인사이트)")
    lines.append("업종 Top5 (표본 30건 이상 중 위험 비율 최고):")
    for name, row in top_fcob.iterrows():
        lines.append(f"  - {name}: {row['mean']*100:.1f}% (n={int(row['count']):,})")
    lines.append("시도 Top5:")
    for name, row in top_region.iterrows():
        lines.append(f"  - {name}: {row['mean']*100:.1f}% (n={int(row['count']):,})")
    lines.append(f"- ![위험 비율]({fig_paths['risk_rate'].relative_to(ROOT).as_posix()})")
    lines.append("")
    lines.append("## 5~6. 수치형 피처")
    age_med_normal = df.loc[df["risk_binary"] == 0, "facility_age_years"].median()
    age_med_risk = df.loc[df["risk_binary"] == 1, "facility_age_years"].median()
    days_med_normal = df.loc[df["risk_binary"] == 0, "days_since_inspection"].median()
    days_med_risk = df.loc[df["risk_binary"] == 1, "days_since_inspection"].median()
    lines.append(f"- 시설연령 중앙값: 정상 {age_med_normal:.1f}년 vs 위험군 {age_med_risk:.1f}년")
    lines.append(f"- 점검경과일수 중앙값: 정상 {days_med_normal:.0f}일 vs 위험군 {days_med_risk:.0f}일")
    lines.append(f"- ![수치형 분포]({fig_paths['numeric'].relative_to(ROOT).as_posix()})")
    lines.append(f"- ![수치형 vs 타겟]({fig_paths['numeric_vs_target'].relative_to(ROOT).as_posix()})")
    lines.append("")
    lines.append("## 7. 결측치 패턴")
    lines.append(f"- ![결측치 패턴]({fig_paths['missing'].relative_to(ROOT).as_posix()})")
    lines.append("")
    lines.append("## 8. 상관관계")
    lines.append(f"- ![상관관계]({fig_paths['correlation'].relative_to(ROOT).as_posix()})")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def run_all():
    df_raw = load_raw()
    df = clean(df_raw)

    fig_paths = {
        "overview": fig_overview(df),
        "target": fig_target(df),
        "categorical": fig_categorical(df),
        "risk_rate": fig_risk_rate(df),
        "numeric": fig_numeric(df),
        "numeric_vs_target": fig_numeric_vs_target(df),
        "missing": fig_missing_pattern(df),
        "correlation": fig_correlation(df),
    }
    report = write_findings_report(df, fig_paths)
    print(f"그림 {len(fig_paths)}개 저장 완료: {FIG_DIR}")
    print(f"요약 리포트 저장 완료: {report}")
    return df, fig_paths


if __name__ == "__main__":
    run_all()
