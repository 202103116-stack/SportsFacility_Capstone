"""
검증 하네스: 위험군 x 점검트랙 층화 K-Fold, OOF 예측, 트랙별 지표

docs/WORKFLOW.md §4 설계 구현. 인코딩·결측 대체는 모두 Pipeline 안에 있어 fold 안에서만 학습된다.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

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


class HGBModel:
    """범주형을 네이티브로 다루는 HistGradientBoosting. 범주 목록은 학습 fold에서만 만든다."""

    def __init__(self, cats=CAT_FEATURES, nums=LINEAR_NUM_FEATURES, **params):
        self.cats, self.nums, self.params = list(cats), list(nums), params

    def _prep(self, X):
        cols = {}
        for c in self.cats:
            known = X[c].where(X[c].isin(self.levels_[c]))  # 학습에 없던 범주는 결측으로
            cols[c] = pd.Categorical(known, categories=self.levels_[c])
        out = pd.DataFrame(cols, index=X.index)
        for c in self.nums:
            out[c] = X[c].to_numpy()
        return out

    def fit(self, X, y):
        self.levels_ = {c: sorted(X[c].dropna().unique()) for c in self.cats}
        self.model_ = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0,
                                                     **self.params).fit(self._prep(X), y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))


def build_forest():
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="미상")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30)),
    ])
    pre = ColumnTransformer([("cat", cat, list(CAT_FEATURES)),
                             ("num", SimpleImputer(strategy="median"), LINEAR_NUM_FEATURES)])
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=0))])


def sample_hgb_configs(n=12, seed=0):
    rng = np.random.default_rng(seed)
    space = {"learning_rate": [0.03, 0.06, 0.1], "max_leaf_nodes": [8, 15, 31, 63],
             "min_samples_leaf": [20, 50, 100, 200], "l2_regularization": [0.0, 1.0, 10.0],
             "class_weight": [None, "balanced"]}
    configs = []
    while len(configs) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in space.items()}
        if cfg not in configs:
            configs.append(cfg)
    base = {"max_iter": 500, "early_stopping": True, "n_iter_no_change": 20, "validation_fraction": 0.15}
    return [{**base, **c} for c in configs]


def macro_track_ap(y, score, track):
    return float(np.mean([average_precision_score(y[track == t], score[track == t]) for t in (0, 1)]))


def hgb_factory(cfg, cats):
    return HGBModel(cats=cats, **cfg)


def select_config(df, y, track, train_idx, configs, seed, inner_splits=3, cats=CAT_FEATURES, factory=hgb_factory):
    """학습 fold 안에서만 3-fold 내부 검증으로 하이퍼파라미터를 고른다(중첩 검증). 목표는 두 트랙 PR-AUC의 평균."""
    d = df.iloc[train_idx].reset_index(drop=True)
    yy, tt = y[train_idx], track[train_idx]
    folds = list(StratifiedKFold(inner_splits, shuffle=True, random_state=seed).split(d, make_strata(yy, tt)))
    scores = []
    for cfg in configs:
        oof = np.zeros(len(d))
        for tr, te in folds:
            oof[te] = factory(cfg, cats).fit(d.iloc[tr], yy[tr]).predict_proba(d.iloc[te])[:, 1]
        scores.append(macro_track_ap(yy, oof, tt))
    return int(np.argmax(scores))


def nested_oof_hgb(df, y, track, configs, seed, n_outer=5, cats=CAT_FEATURES, factory=hgb_factory):
    outer = list(StratifiedKFold(n_outer, shuffle=True, random_state=seed).split(df, make_strata(y, track)))

    def work(tr, te):
        with threadpool_limits(limits=1):
            best = select_config(df, y, track, tr, configs, seed + 1000, cats=cats, factory=factory)
            p = factory(configs[best], cats).fit(df.iloc[tr], y[tr]).predict_proba(df.iloc[te])[:, 1]
        return te, p, best

    oof, chosen = np.zeros(len(df)), []
    for te, p, best in Parallel(n_jobs=n_outer)(delayed(work)(tr, te) for tr, te in outer):
        oof[te] = p
        chosen.append(best)
    return oof, chosen


def oof_grouped(make_model, df, y, strata, groups, n_splits=5, seed=42):
    oof = np.zeros(len(df))
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(df, strata, groups):
        oof[te] = make_model().fit(df.iloc[tr], y[tr]).predict_proba(df.iloc[te])[:, 1]
    return oof


class BoostModel:
    """LightGBM / CatBoost 래퍼. 학습 fold 안에서 15%를 떼어 조기 종료에 쓴다(HGB 내부 검증과 같은 비율)."""

    def __init__(self, kind, cats=CAT_FEATURES, nums=LINEAR_NUM_FEATURES, **params):
        self.kind, self.cats, self.nums, self.params = kind, list(cats), list(nums), params

    def _prep(self, X):
        if self.kind == "lightgbm":
            cols = {c: pd.Categorical(X[c].where(X[c].isin(self.levels_[c])), categories=self.levels_[c]) for c in self.cats}
            out = pd.DataFrame(cols, index=X.index)
        else:
            out = pd.DataFrame({c: X[c].fillna("미상").astype(str) for c in self.cats}, index=X.index)
        for c in self.nums:
            out[c] = X[c].to_numpy()
        return out

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        self.levels_ = {c: sorted(X[c].dropna().unique()) for c in self.cats}
        tr, va = train_test_split(np.arange(len(X)), test_size=0.15, stratify=y, random_state=0)
        Xp = self._prep(X)
        if self.kind == "lightgbm":
            from lightgbm import LGBMClassifier, early_stopping
            self.model_ = LGBMClassifier(n_estimators=1000, random_state=0, n_jobs=1, verbosity=-1, **self.params)
            self.model_.fit(Xp.iloc[tr], y[tr], eval_set=[(Xp.iloc[va], y[va])],
                            callbacks=[early_stopping(30, verbose=False)])
        else:
            from catboost import CatBoostClassifier
            self.model_ = CatBoostClassifier(iterations=1000, random_seed=0, thread_count=1, verbose=0,
                                             cat_features=self.cats, early_stopping_rounds=30, **self.params)
            self.model_.fit(Xp.iloc[tr], y[tr], eval_set=(Xp.iloc[va], y[va]))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))


def lightgbm_factory(cfg, cats):
    return BoostModel("lightgbm", cats=cats, **cfg)


def catboost_factory(cfg, cats):
    return BoostModel("catboost", cats=cats, **cfg)


def _sample_configs(space, n, seed):
    rng = np.random.default_rng(seed)
    configs = []
    while len(configs) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in space.items()}
        if cfg not in configs:
            configs.append(cfg)
    return configs


def sample_lightgbm_configs(n=12, seed=0):
    return _sample_configs({"learning_rate": [0.03, 0.06, 0.1], "num_leaves": [8, 15, 31, 63],
                            "min_child_samples": [20, 50, 100, 200], "reg_lambda": [0.0, 1.0, 10.0],
                            "cat_smooth": [10, 50], "class_weight": [None, "balanced"]}, n, seed)


def sample_catboost_configs(n=6, seed=0):
    # 학습 1회가 ~17초라 탐색 범위를 좁힘(깊이 8, 학습률 0.03 제외)
    return _sample_configs({"learning_rate": [0.06, 0.1], "depth": [4, 6], "l2_leaf_reg": [1, 3, 10],
                            "auto_class_weights": [None, "Balanced"]}, n, seed)
