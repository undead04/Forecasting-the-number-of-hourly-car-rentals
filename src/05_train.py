"""Giai đoạn 5 — Train model.
Input : datas/04_split/{X_train_*, y_train_*} + selected_features.json
Output: datas/05_models/{best_casual.pkl, best_registered.pkl, best_cnt.pkl, cv_results_*.csv}
- Prep: Linear = Poly(1-2)+Scale+OneHot ; Tree = passthrough+Ordinal
- GridSearchCV + TimeSeriesSplit(5), scoring neg_MAE
- Train 2 target riêng + 1 baseline cnt trực tiếp
Chạy: python src/05_train.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from config import SPLIT_DIR, MODELS_DIR, RANDOM_STATE


def get_prep(num_cols, cat_cols):
    num_pipe = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                         ("scaler", StandardScaler())])
    prep_line = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])
    prep_tree = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)])
    return prep_line, prep_tree


def get_param_grids(prep_line, prep_tree):
    return {
        "PolyLinear": {"model": Pipeline([("prep", prep_line), ("ridge", Ridge())]),
                       "params": {"prep__num__poly__degree": [1, 2],
                                  "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}},
        "DecisionTree": {"model": Pipeline([("prep", prep_tree),
                                            ("tree", DecisionTreeRegressor(random_state=RANDOM_STATE))]),
                         "params": {"tree__max_depth": [3, 5, 8, 12, None],
                                    "tree__min_samples_leaf": [1, 5, 10, 20],
                                    "tree__max_features": ["sqrt", 0.5, 1.0]}},
        "RandomForest": {"model": Pipeline([("prep", prep_tree),
                                            ("rf", RandomForestRegressor(
                                                random_state=RANDOM_STATE, n_jobs=-1))]),
                         "params": {"rf__n_estimators": [100, 300],
                                    "rf__max_depth": [5, 10, 20, None],
                                    "rf__max_features": [0.3, 0.5, "sqrt"],
                                    "rf__min_samples_leaf": [1, 5, 10]}},
    }


def train_one(X, y, tag):
    sel = json.loads((SPLIT_DIR / "selected_features.json").read_text(encoding="utf-8"))
    if tag == "cnt":
        num_c = [c for c in sel["num_casual"] + sel["num_registered"] if c in X.columns]
        cat_c = [c for c in sel["cat_casual"] + sel["cat_registered"] if c in X.columns]
    else:
        num_c, cat_c = sel[f"num_{tag}"], sel[f"cat_{tag}"]
    prep_line, prep_tree = get_prep(num_c, cat_c)
    kf = TimeSeriesSplit(n_splits=5)
    rows, best = [], None
    for name, cfg in get_param_grids(prep_line, prep_tree).items():
        gs = GridSearchCV(cfg["model"], cfg["params"], cv=kf,
                          scoring="neg_mean_absolute_error", n_jobs=-1)
        gs.fit(X, y.values.ravel())
        # CV bổ sung RMSE/R2 trên cùng fold
        rms, r2s, maes = [], [], []
        for tr, va in kf.split(X):
            m = clone(gs.best_estimator_).fit(X.iloc[tr], y.values.ravel()[tr])
            p = m.predict(X.iloc[va])
            maes.append(mean_absolute_error(y.values.ravel()[va], p))
            rms.append(root_mean_squared_error(y.values.ravel()[va], p))
            r2s.append(r2_score(y.values.ravel()[va], p))
        rows.append({"model": name, "mae_mean": -gs.best_score_,
                     "mae_std": float(np.std(maes)),
                     "rmse_mean": float(np.mean(rms)),
                     "r2_mean": float(np.mean(r2s)),
                     "best_params": json.dumps(gs.best_params_, default=str)})
        if best is None or rows[-1]["mae_mean"] < best[1]:
            best = (gs.best_estimator_, rows[-1]["mae_mean"])
        print(f"[{tag}] {name}: MAE={rows[-1]['mae_mean']:.4f}")
    pd.DataFrame(rows).sort_values("mae_mean").to_csv(
        MODELS_DIR / f"cv_results_{tag}.csv", index=False)
    joblib.dump(best[0], MODELS_DIR / f"best_{tag}.pkl")
    return best[0]


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for tag in ["casual", "registered", "cnt"]:
        X = pd.read_csv(SPLIT_DIR / f"X_train_{tag}.csv")
        y = pd.read_csv(SPLIT_DIR / f"y_train_{tag}.csv")
        train_one(X, y, tag)
        print(f"OK best_{tag}.pkl")
    print(f"OK -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
