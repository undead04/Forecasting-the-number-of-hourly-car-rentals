"""Giai đoạn 6 — Đánh giá kết quả train.
Input : datas/05_models/best_*.pkl + datas/04_split/{X_test_*, y_test_*}
Output: datas/06_evaluation/{metrics.csv, predictions.csv, compare_baseline.csv} + figures/
- Dự báo clip >= 0 (count), metric R2/RMSE/RMSLE/MAPE (mask y_true != 0, không dùng biến global)
- So sánh (casual+registered) vs baseline cnt trực tiếp + plot sorted & residual
Chạy: python src/06_evaluate.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from config import SPLIT_DIR, MODELS_DIR, EVAL_DIR

FIG = EVAL_DIR / "figures"


def test_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.clip(np.asarray(y_pred).flatten(), 0, None)
    mask = y_true != 0
    mape = (mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
            if mask.sum() else float("nan"))
    return {"R2": round(float(r2_score(y_true, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "RMSLE": round(float(np.sqrt(mean_squared_error(
                np.log1p(y_true), np.log1p(y_pred)))), 4),
            "MAPE": round(float(mape), 2)}


def plot_sorted(y_true, y_pred, name):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.clip(np.asarray(y_pred).flatten(), 0, None)
    idx = np.argsort(y_true)
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[idx], label="true", linewidth=1)
    plt.plot(y_pred[idx], label="pred", linewidth=1)
    plt.title(f"Sorted pred vs true — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"sorted_{name}.png", dpi=120)
    plt.close()


def plot_residual(y_true, y_pred, name):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.clip(np.asarray(y_pred).flatten(), 0, None)
    res = y_true - y_pred
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(x=y_pred, y=res, alpha=0.3, ax=ax[0])
    ax[0].axhline(0, color="red")
    ax[0].set_title(f"Residual vs pred — {name}")
    sns.histplot(res, kde=True, ax=ax[1])
    ax[1].set_title(f"Residual dist — {name}")
    plt.tight_layout()
    plt.savefig(FIG / f"residual_{name}.png", dpi=120)
    plt.close()


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    best_c = joblib.load(MODELS_DIR / "best_casual.pkl")
    best_r = joblib.load(MODELS_DIR / "best_registered.pkl")
    best_cnt = joblib.load(MODELS_DIR / "best_cnt.pkl")

    out, preds = [], {}
    for tag, model in [("casual", best_c), ("registered", best_r), ("cnt", best_cnt)]:
        Xt = pd.read_csv(SPLIT_DIR / f"X_test_{tag}.csv")
        yt = pd.read_csv(SPLIT_DIR / f"y_test_{tag}.csv").values.ravel()
        yp = np.clip(model.predict(Xt), 0, None)
        m = {"target": tag, **test_metrics(yt, yp)}
        out.append(m)
        preds[tag] = (yt, yp)
        plot_sorted(yt, yp, tag)
        plot_residual(yt, yp, tag)

    # Cộng 2 mô hình so với baseline cnt trực tiếp (chung index thời gian)
    yc = preds["casual"][0] + preds["registered"][0]
    yp_sum = np.clip(preds["casual"][1] + preds["registered"][1], 0, None)
    out.append({"target": "sum_casual_registered", **test_metrics(yc, yp_sum)})
    plot_sorted(yc, yp_sum, "sum_casual_registered")
    plot_residual(yc, yp_sum, "sum_casual_registered")

    pd.DataFrame(out).to_csv(EVAL_DIR / "metrics.csv", index=False)
    pd.DataFrame({"y_true_sum": yc, "y_pred_sum": yp_sum,
                  "y_true_cnt": preds["cnt"][0],
                  "y_pred_cnt": preds["cnt"][1]}).to_csv(
        EVAL_DIR / "predictions.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False))
    print(f"OK -> {EVAL_DIR}")


if __name__ == "__main__":
    main()
