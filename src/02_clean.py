"""Giai đoạn 2 — Làm sạch & chuẩn hóa.
Input : datas/hour.csv (raw)
Output: datas/02_cleaned/cleaned.csv + dtypes.json + cleaning_report.json
- dteday str -> datetime, sort theo (dteday, hr)
- 8 cột rời rạc -> category (GIỮ category, không astype int trở lại)
- Không xóa dòng (raw không null); chỉ báo cáo null/duplicates/outlier-nhẹ
Chạy: python src/02_clean.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import RAW_CSV, CLEANED_DIR

CAT8 = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    report = {"n_raw": len(df)}

    df["dteday"] = pd.to_datetime(df["dteday"])
    for c in CAT8:
        if c in df.columns:
            df[c] = df[c].astype("category")

    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    report.update({
        "n_cleaned": len(df),
        "null_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "date_min": str(df["dteday"].min()),
        "date_max": str(df["dteday"].max()),
    })

    df.to_csv(CLEANED_DIR / "cleaned.csv", index=False)
    (CLEANED_DIR / "dtypes.json").write_text(
        df.dtypes.astype(str).to_json(indent=2), encoding="utf-8")
    (CLEANED_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"OK -> {CLEANED_DIR / 'cleaned.csv'}")


if __name__ == "__main__":
    main()
