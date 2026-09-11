# Changelog

Mọi thay đổi đáng chú ý của project được ghi lại tại đây.

## [Unreleased] — nhánh `danh`

### Added
- Pipeline 6 notebook giai đoạn trong `src/` (`01_eda → 06_evaluate` + `run_all.ipynb`), mỗi giai đoạn lưu output riêng vào `datas/01_eda → 06_evaluation`.
- Features mới: lag/rolling (`lag1/lag24`, `roll3/roll24`), `comfort`, `is_bad_weather`, cyclic sin/cos.
- Models mới: Ridge-log (log1p-target), HistGradientBoosting (early stopping), Poisson, Tweedie(1.5); `RandomizedSearchCV` cho RF/HGB.
- Đánh giá mới: hiệu chỉnh count (clip + round), baseline `cnt` trực tiếp, metrics theo phân khúc (`segment_metrics.csv`), VIF, lưu `best_*.pkl`.
- `danh_gia_project.md` (đánh giá + cải tiến), `requirements.txt` gọn (bỏ metapackage `jupyter` → `ipykernel`+`nbclient`).
- Auto-save figures của `do_an_2.ipynb` vào `output/`.

### Fixed
- Bỏ leakage: rush-hour dò từ full target, ANOVA/MI trên full data, split random/KFold-shuffle trên time-series → split thời gian + `TimeSeriesSplit` + selection train-only.
- Bỏ cell `astype("int")` phá `category`; khai báo cat/num tường minh.
- Bug `evaluate_model` (mask biến global), `get_best_model` (sort + rename cột sai), cộng 2 target khác index.
- Giảm đa cộng tuyến: bỏ `atemp`, bỏ `temp_squared/hum_squared` trùng `Poly`.
- Đồng bộ `README.md` sang dataset UCI + pipeline `src/`.

### Changed
- `do_an_2.ipynb`: giữ nguyên EDA, áp dụng các fix P0 (chưa re-run toàn bộ sau fix).

## [2026-09-11] — remote `origin/main` (Trần Văn An)

- Pipeline 4 notebook trong `notebooks/` + model RF (`notebooks/models/best_*_rf.joblib`), splits ở `datas/` root.
- Xóa `datas/SeoulBikeData.csv`, viết lại `README.md`.
- Metrics RF: tổng R² 0.902 / RMSE 68.9 (thua pipeline `src/`).
- ⚠️ Chờ thống nhất cấu trúc (`src/` vs `notebooks/`) trước khi merge.

## [2026-09-09] — split code

- Tách pipeline từ notebook sang `src/*.py` (sau chuyển thành `.ipynb`).

## [2026-09-07] — Notebook version 2

- `do_an_2.ipynb`: EDA + train Ridge/Tree/RF trên `datas/hour.csv` (UCI), 2-model `casual`+`registered`.

## [2026-08-28] — Khởi tạo

- First commit; `README.md` mô tả dataset Seoul; xóa `.idea`.
