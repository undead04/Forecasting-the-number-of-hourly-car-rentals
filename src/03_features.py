"""Giai đoạn 3 — Feature engineering (bản FIX, không leakage).
Input : datas/02_cleaned/cleaned.csv
Output: datas/03_features/features.csv + feature_list.json
- is_weekend, cyclic hr/mnth/weekday (sin/cos)
- rush theo DOMAIN (sáng 7-9, chiều 17-19, trưa 12-14), KHÔNG dò peak từ target
- time_period (Night/Morning/Afternoon/Evening/Late Night)
- temp_hum_interaction (không tạo temp_squared/hum_squared vì Poly degree=2 đã sinh)
- Bỏ atemp (cộng tuyến ~0.99 với temp)
Chạy: python src/03_features.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from config import CLEANED_DIR, FEATURES_DIR, CAT_COLS


def get_time_period(hr: int) -> str:
    hr = int(hr)
    if 0 <= hr <= 5:
        return "Night"
    if 6 <= hr <= 11:
        return "Morning"
    if 12 <= hr <= 16:
        return "Afternoon"
    if 17 <= hr <= 21:
        return "Evening"
    return "Late Night"


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_DIR / "cleaned.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])

    df["is_weekend"] = df["weekday"].astype(int).isin([0, 6]).astype(int)

    df["hr_sin"] = np.sin(2 * np.pi * df["hr"].astype(int) / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"].astype(int) / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"].astype(int) / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"].astype(int) / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"].astype(int) / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"].astype(int) / 7)

    df["morning_rush"] = df["hr"].astype(int).isin([7, 8, 9]).astype(int)
    df["evening_rush"] = df["hr"].astype(int).isin([17, 18, 19]).astype(int)
    df["midday_rest"] = df["hr"].astype(int).isin([12, 13, 14]).astype(int)
    wd = df["workingday"].astype(int)
    df["workingday_morning_rush"] = ((wd == 1) & (df["morning_rush"] == 1)).astype(int)
    df["workingday_evening_rush"] = ((wd == 1) & (df["evening_rush"] == 1)).astype(int)
    df["non_workingday_midday"] = ((wd == 0) & (df["midday_rest"] == 1)).astype(int)

    df["time_period"] = df["hr"].apply(get_time_period).astype("category")
    df["temp_hum_interaction"] = df["temp"] * df["hum"]
    if "atemp" in df.columns:  # giảm đa cộng tuyến temp/atemp
        df = df.drop(columns=["atemp"])

    for c in CAT_COLS:  # đảm bảo category được giữ
        if c in df.columns:
            df[c] = df[c].astype("category")

    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ("instant", "cnt", "casual", "registered")]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    df.to_csv(FEATURES_DIR / "features.csv", index=False)
    (FEATURES_DIR / "feature_list.json").write_text(json.dumps(
        {"cat_cols": cat_cols, "num_cols": num_cols}, indent=2), encoding="utf-8")
    print(f"cat: {cat_cols}\nnum: {num_cols}")
    print(f"OK -> {FEATURES_DIR / 'features.csv'} ({len(df)} dòng)")


if __name__ == "__main__":
    main()
