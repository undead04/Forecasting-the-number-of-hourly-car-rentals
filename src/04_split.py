"""Giai đoạn 4 — Split train/test + chọn đặc trưng (train-only, không leakage).
Input : datas/03_features/features.csv
Output: datas/04_split/{X_train_*, X_test_*, y_train_*, y_test_*, selected_features.json, anova_*.csv, mi_*.csv, split_info.json}
- Split theo THỜI GIAN 80/20 (quá khứ -> train), 1 bộ index chung cho cả 3 target
- ANOVA (categorical) + Mutual Information (numeric) chỉ fit trên TRAIN
Chạy: python src/04_split.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from config import FEATURES_DIR, SPLIT_DIR, TEST_SIZE


def anova_select(df_train, cat_cols, target, alpha=0.05):
    rows, selected = [], []
    for col in cat_cols:
        groups = [g[target].values for _, g in df_train.groupby(col, observed=True)]
        if len(groups) <= 1:
            continue
        f, p = stats.f_oneway(*groups)
        ok = bool(p < alpha)
        if ok:
            selected.append(col)
        rows.append({"feature": col, "F": round(float(f), 4),
                     "p_value": float(p), "selected": ok})
    out = pd.DataFrame(rows).sort_values("F", ascending=False) if rows else pd.DataFrame()
    return selected, out


def mi_select(df_train, num_cols, target, threshold=0.05):
    feats = [c for c in num_cols if c not in ("casual", "registered", "cnt")]
    X = df_train[feats].copy()
    mask = [pd.api.types.is_integer_dtype(X[c]) or
            isinstance(X[c].dtype, pd.CategoricalDtype) for c in feats]
    scores = mutual_info_regression(X, df_train[target], discrete_features=mask, random_state=42)
    rows, selected = [], []
    for c, s in zip(feats, scores):
        ok = bool(s >= threshold)
        if ok:
            selected.append(c)
        rows.append({"feature": c, "mi": round(float(s), 4), "selected": ok})
    return selected, pd.DataFrame(rows).sort_values("mi", ascending=False)


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_DIR / "features.csv", parse_dates=["dteday"])
    fl = json.loads((FEATURES_DIR / "feature_list.json").read_text(encoding="utf-8"))
    cat_cols, num_cols = fl["cat_cols"], fl["num_cols"]

    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    k = int(len(df) * (1 - TEST_SIZE))
    train_idx, test_idx = df.index[:k], df.index[k:]
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]

    out = {}
    for target in ["casual", "registered"]:
        sc, anova_df = anova_select(df_train, cat_cols, target)
        sn, mi_df = mi_select(df_train, num_cols, target)
        out[f"cat_{target}"] = sc
        out[f"num_{target}"] = sn
        if not anova_df.empty:
            anova_df.to_csv(SPLIT_DIR / f"anova_{target}.csv", index=False)
        mi_df.to_csv(SPLIT_DIR / f"mi_{target}.csv", index=False)

    feats_cnt = sorted(set(out["cat_casual"] + out["num_casual"] +
                           out["cat_registered"] + out["num_registered"]))
    out["baseline_cnt_features"] = feats_cnt

    # Lưu split theo từng target, chung index thời gian
    for target in ["casual", "registered"]:
        feats = out[f"cat_{target}"] + out[f"num_{target}"]
        df[feats].loc[train_idx].to_csv(SPLIT_DIR / f"X_train_{target}.csv", index=False)
        df[feats].loc[test_idx].to_csv(SPLIT_DIR / f"X_test_{target}.csv", index=False)
        df[[target]].loc[train_idx].to_csv(SPLIT_DIR / f"y_train_{target}.csv", index=False)
        df[[target]].loc[test_idx].to_csv(SPLIT_DIR / f"y_test_{target}.csv", index=False)
    df[feats_cnt].loc[train_idx].to_csv(SPLIT_DIR / "X_train_cnt.csv", index=False)
    df[feats_cnt].loc[test_idx].to_csv(SPLIT_DIR / "X_test_cnt.csv", index=False)
    df[["cnt"]].loc[train_idx].to_csv(SPLIT_DIR / "y_train_cnt.csv", index=False)
    df[["cnt"]].loc[test_idx].to_csv(SPLIT_DIR / "y_test_cnt.csv", index=False)

    (SPLIT_DIR / "selected_features.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    (SPLIT_DIR / "split_info.json").write_text(json.dumps({
        "n_train": len(train_idx), "n_test": len(test_idx),
        "train_range": [str(df_train["dteday"].min()), str(df_train["dteday"].max())],
        "test_range": [str(df_test["dteday"].min()), str(df_test["dteday"].max())],
    }, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"OK -> {SPLIT_DIR}")


if __name__ == "__main__":
    main()
