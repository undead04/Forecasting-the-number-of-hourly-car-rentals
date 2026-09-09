"""Giai đoạn 1 — Khám phá dữ liệu (EDA).
Đọc raw, chỉ phân tích + lưu báo cáo, KHÔNG sửa df.
Input : datas/hour.csv
Output: datas/01_eda/{describe, null_report, skew_kurtosis, outlier_report, spearman_corr}.csv + figures/
Chạy: python src/01_eda.py
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
from scipy import stats
from config import RAW_CSV, EDA_DIR

FIG = EDA_DIR / "figures"


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    print(f"shape: {df.shape}")

    # 1. describe số
    df.describe().to_csv(EDA_DIR / "describe.csv")

    # 2. null report
    null_df = pd.DataFrame({
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "dtype": df.dtypes.astype(str),
    })
    null_df.to_csv(EDA_DIR / "null_report.csv")
    print("null tổng:", int(null_df["null_count"].sum()),
          "| duplicates:", int(df.duplicated().sum()))

    # 3. skew / kurtosis cho cột số
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = [{"feature": c, "skew": round(df[c].skew(), 3),
             "kurtosis": round(df[c].kurtosis(), 3)} for c in num]
    pd.DataFrame(rows).to_csv(EDA_DIR / "skew_kurtosis.csv", index=False)

    # 4. outlier IQR vs Z-score
    rep = []
    for c in num:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        iqr_n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
        z = np.abs(stats.zscore(df[c].dropna()))
        rep.append({"feature": c, "iqr_n": int(iqr_n),
                    "iqr_pct": round(iqr_n / len(df) * 100, 2),
                    "z_n": int((z > 3).sum())})
    pd.DataFrame(rep).to_csv(EDA_DIR / "outlier_report.csv", index=False)

    # 5. tương quan Spearman (phi tuyến) cho cột số
    corr = df[num].corr(method="spearman")
    corr.to_csv(EDA_DIR / "spearman_corr.csv")
    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman correlation (numeric)")
    plt.tight_layout()
    plt.savefig(FIG / "spearman_heatmap.png", dpi=120)
    plt.close()

    # 6. phân phối 2 target
    for t in ["casual", "registered", "cnt"]:
        if t not in df.columns:
            continue
        plt.figure(figsize=(8, 4))
        sns.histplot(df[t], kde=True)
        plt.title(f"Distribution of {t}")
        plt.tight_layout()
        plt.savefig(FIG / f"dist_{t}.png", dpi=120)
        plt.close()

    print(f"OK -> {EDA_DIR}")


if __name__ == "__main__":
    main()
