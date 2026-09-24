"""
검증 하네스: 위험군 x 점검트랙 층화 K-Fold, OOF 예측, 트랙별 지표

docs/WORKFLOW.md §4 설계 구현. 인코딩·결측 대체는 모두 Pipeline 안에 있어 fold 안에서만 학습된다.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import CAT_FEATURES, OUT_PATH

LINEAR_NUM_FEATURES = ["log_area", "age_years"]
GROUPS = ["전체", "공식점검", "자율점검"]


def load_labeled():
    df = pd.read_csv(OUT_PATH, low_memory=False)
    df = df[df["has_label"]].reset_index(drop=True)
    y = df["risk_binary"].astype(int).to_numpy()
    track = (df["inspection_track"] == "자율점검").astype(int).to_numpy()
    return df, y, track


def make_strata(y, track):
    return y * 2 + track


def group_mask(track, group):
    if group == "전체":
        return np.ones(len(track), dtype=bool)
    return track == (1 if group == "자율점검" else 0)


def build_logistic(add_missing_flags=False, C=1.0, cat_features=CAT_FEATURES, num_features=LINEAR_NUM_FEATURES):
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="미상")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30)),
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=add_missing_flags)),
        ("scale", StandardScaler()),
    ])
    transformers = [("cat", cat, list(cat_features))] if cat_features else []
    if num_features:
        transformers.append(("num", num, list(num_features)))
    return Pipeline([("pre", ColumnTransformer(transformers)),
                     ("clf", LogisticRegression(class_weight="balanced", C=C, max_iter=2000))])


class CategoryRateBaseline:
    """한 범주형 컬럼의 (평활화된) 위험군 비율을 그대로 점수로 쓰는 단순 기준선."""

    def __init__(self, col="ftype_nm", prior_strength=20.0):
        self.col = col
        self.prior_strength = prior_strength

    def fit(self, X, y):
        self.prior_ = float(np.mean(y))
        g = pd.DataFrame({"c": X[self.col].fillna("미상").to_numpy(), "y": y}).groupby("c")["y"].agg(["sum", "count"])
        self.rate_ = (g["sum"] + self.prior_strength * self.prior_) / (g["count"] + self.prior_strength)
        return self

    def predict_proba(self, X):
        p = X[self.col].fillna("미상").map(self.rate_).fillna(self.prior_).to_numpy()
        return np.column_stack([1 - p, p])


def oof_predictions(make_model, df, y, strata, n_splits=5, seed=42):
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, strata):
        model = make_model().fit(df.iloc[tr], y[tr])
        oof[te] = model.predict_proba(df.iloc[te])[:, 1]
    return oof


def group_metrics(y, score, mask, top_ks=(0.05, 0.10, 0.20), seed=0):
    yy, ss = y[mask], score[mask]
    base = yy.mean()
    ap = average_precision_score(yy, ss)
    out = {"n": int(len(yy)), "위험군": int(yy.sum()), "기준선": base, "PR-AUC": ap,
           "PR-AUC 배수": ap / base, "ROC-AUC": roc_auc_score(yy, ss)}
    # 동점(범주형 기준선 등)이 행 순서에 영향받지 않도록 무작위 순서에서 안정 정렬
    perm = np.random.default_rng(seed).permutation(len(ss))
    order = perm[np.argsort(-ss[perm], kind="stable")]
    for k in top_ks:
        m = max(1, int(round(len(yy) * k)))
        hit = yy[order[:m]].sum()
        out[f"Recall@{int(k * 100)}%"] = hit / yy.sum()
        out[f"Lift@{int(k * 100)}%"] = (hit / m) / base
    return out


def bootstrap_ap_ci(y, score, mask, n_boot=200, seed=0):
    yy, ss = y[mask], score[mask]
    rng = np.random.default_rng(seed)
    vals = [average_precision_score(yy[i], ss[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(n_boot))]
    return tuple(np.percentile(vals, [2.5, 97.5]))
