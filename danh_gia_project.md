# Đánh giá chi tiết project — Dự đoán số lượt thuê xe theo giờ

> Nguồn rà soát: `do_an_2.ipynb` (147 cells), `datas/hour.csv`, `datas/SeoulBikeData.csv`, `README.md`
> Ngày đánh giá: 2026-09-09
> Cập nhật cải tiến: 2026-09-11 (pipeline `src/`, xem mục 6)
> Phạm vi: tiền xử lý dữ liệu, feature engineering, chọn đặc trưng, huấn luyện và đánh giá mô hình.

---

## 1. Tổng quan project

- **Bài toán:** Supervised Learning — Regression, dự đoán nhu cầu thuê xe theo giờ.
- **Dataset thực dùng:** `datas/hour.csv` (UCI Bike Sharing, Capital Bikeshare 2011–2012), ~17.379 dòng × 17 cột:
  `instant, dteday, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, cnt` với `cnt = casual + registered`.
- **Cách tiếp cận ban đầu:** không dự đoán trực tiếp `cnt`, mà train 2 mô hình riêng cho `casual` và `registered`, sau đó cộng lại: `y_pred = y_pred_casual + y_pred_registered`.
- **Pipeline trong notebook:** Data Loading → Data Understanding → EDA → Feature Engineering → Feature Selection (ANOVA + MI) → Train (Ridge / DecisionTree / RandomForest + GridSearchCV 5-fold) → Đánh giá test (R², RMSE, RMSLE, MAPE) + phân tích phần dư.
- **File thừa / lệch tài liệu:** `datas/SeoulBikeData.csv` không được dùng; `README.md` (bản cũ) vẫn mô tả dataset Seoul và file `do_an.ipynb`, không khớp `do_an_2.ipynb` + `hour.csv`.

---

## 2. Các bước tiền xử lý dữ liệu đã thực hiện (bản notebook gốc)

| # | Bước | Vị trí | Nội dung cụ thể | Kết quả |
|---|------|--------|-----------------|---------|
| 1 | Load dữ liệu | cell 5 | `pd.read_csv("./datas/hour.csv")` | OK |
| 2 | Ép kiểu thời gian | cell 12 | `dteday: str → datetime` | OK, nhưng sau đó không khai thác `dteday` (không tạo lag/trend) |
| 3 | Ép kiểu phân loại | cell 13 | `season, yr, mnth, hr, holiday, weekday, workingday, weathersit → category` | Đúng ý đồ, nhưng bị đảo ngược ở cell 111 |
| 4 | Kiểm tra tổng quan | cell 6–10, 14 | `shape, info, head/tail, describe` | Đầy đủ |
| 5 | Kiểm tra null | cell 15 | Hàm `check_null()`: đếm null + % + dtype | Kết luận không có null — OK |
| 6 | Phân tích phân phối số | cell 17–18 | skew/kurtosis + hist + KDE + mean/median | Tốt về EDA, nhưng không dẫn tới hành động xử lý |
| 7 | Phát hiện outlier | cell 19–20, 28–29 | So sánh IQR (1.5) vs Z-score (>3) + boxplot | Chỉ báo cáo, không xử lý |
| 8 | EDA thời gian | cell 32–56 | Thuê xe theo `hr, mnth, weekday, season`, theo năm, workingday/holiday, top-3 giờ cao điểm | Rất kỹ, nhưng kết quả bị dùng để hard-code feature → leakage |
| 9 | EDA hành vi user | cell 58–73 | So sánh `casual vs registered` theo giờ/ngày/mùa/workingday/holiday, pie tỷ trọng, ratio theo giờ | Tốt |
| 10 | EDA thời tiết | cell 76–91 | Tạo `temp_actual/hum_actual/windspeed_actual`; regplot, boxplot theo `weathersit`, Pearson, FacetGrid | Tốt, nhưng cột trùng lặp rồi loại bỏ không nhất quán |
| 11 | Correlation | cell 94–108 | Ma trận Spearman, barplot corr, bảng so sánh Casual vs Registered | Hợp lý khi dùng Spearman cho quan hệ phi tuyến |
| 12 | Feature engineering | cell 110 | `is_weekend`, 4 flag rush-hour, `time_period`, `temp_squared`, `hum_squared`, `temp*hum` | Có leakage + đa cộng tuyến |
| 13 | Đảo kiểu category→int | cell 111 | `df[cols].astype("int")` | Sai — phá bước 3 |
| 14 | Tái xác định cat/num | cell 112–113 | `select_dtypes` lại | Hệ quả lỗi cell 111: `cat_cols` thực tế chỉ còn `time_period` |
| 15 | Chọn đặc trưng | cell 114–117 | ANOVA (α=0.05) + Mutual Information (threshold=0.05), tách riêng `casual`/`registered` | Chạy trên full data → leakage |
| 16 | Split train/test | cell 119–120 | `train_test_split(test_size=0.2, random_state=42)` × 2 target | Random split trên time-series → sai phương pháp |
| 17 | Preprocessing cho model | cell 122 | Linear: `Poly → StandardScaler + OneHot`; Tree: `passthrough + Ordinal` | Ý tưởng đúng, nhưng đầu vào cat/num đã sai; Poly + feature bình phương thủ công gây dư thừa |
| 18 | Tuning + CV | cell 123–124, 128 | `GridSearchCV(cv=KFold(5, shuffle=True))`, scoring `neg_MAE` cho Ridge/Tree/RF | Shuffle trên time-series → leakage; thiếu XGB/LGBM như comment hứa |
| 19 | Đánh giá | cell 125–147 | CV MAE/RMSE/R², test R²/RMSE/RMSLE/MAPE, plot sorted pred-vs-true, residual, cộng 2 target thành `cnt` | Có bug biến global + sort/table, cộng Series từ 2 split độc lập rất mong manh |

**Không thực hiện:** kiểm tra duplicate, xử lý outlier/skew (log/Box-Cox), chuẩn hóa target, VIF/đa cộng tuyến, tách thời gian, lưu pipeline/model, baseline so sánh.

---

## 3. Các điểm bất hợp lý và lỗi (bản notebook gốc)

### 3.1. Data leakage từ EDA → Feature Engineering (nghiêm trọng)

- **Bằng chứng:** cell 55–56 tính peak-hours trên toàn bộ dữ liệu, cell 110 hard-code `hr in [16,17,18], [8,17,18]...` làm feature.
- **Vì sao sai:** ngưỡng giờ cao điểm được "nhìn" từ target của cả train + test → MAE/R² lạc quan giả.
- **Sửa:** định nghĩa khung giờ theo domain knowledge, hoặc chỉ thống kê peak trên train.

### 3.2. Leakage trong chọn đặc trưng và split sai cho time-series (nghiêm trọng)

- ANOVA/MI chạy trên full `df` rồi mới split; `train_test_split` random + `KFold(shuffle=True)` khiến "tương lai" (2012) lọt vào train.
- **Sửa:** split theo thời gian trước, `TimeSeriesSplit`, mọi bước fit chỉ trên train/fold-train.

### 3.3. Đảo ngược kiểu dữ liệu categorical

- Cell 111 `astype("int")` xóa `category` của cell 13 → `cat_cols` chỉ còn `time_period`; `hr/mnth/weekday...` bị coi là số liên tục có thứ tự (sai ngữ nghĩa với Ridge + Poly).
- **Sửa:** bỏ cell 111; giữ `category`; cyclic encoding sin/cos cho `hr/mnth/weekday`.

### 3.4. Đa cộng tuyến và feature dư thừa

- Giữ đồng thời `temp` và `atemp` (~0.99); `temp_squared/hum_squared` thủ công + `PolynomialFeatures(degree=2)` tạo trùng lặp; `temp_actual/...` tạo rồi drop không nhất quán.
- **Sửa:** chọn 1 trong `temp/atemp`; bỏ feature bình phương thủ công; chạy VIF.

### 3.5. Bug code trong đánh giá

1. Trùng tên `evaluate_model` (cell 125 CV vs 135 test) → ghi đè.
2. `mask = y_test != 0` dùng biến global thay vì `y_true` → MAPE sai.
3. `get_best_model` sort nhưng dựng bảng từ list chưa sort + gán đè tên cột sai.
4. Cộng 2 target từ 2 split độc lập — mong manh, chỉ đúng nhờ trùng `random_state`.

### 3.6. Thiếu xử lý outlier / skew của target

- `casual/registered/cnt` lệch phải mạnh nhưng không transform/clip; dự báo có thể âm.
- **Sửa:** `log1p`-target hoặc Poisson/Tweedie; `clip(>=0)`.

### 3.7. Thiết kế 2 mô hình + tập model hẹp

- Không có baseline `cnt` trực tiếp để chứng minh 2-model đáng giá; comment ghi XGB/LGBM nhưng chỉ có Ridge/Tree/RF.

### 3.8. Tài liệu và tái lập

- README mô tả Seoul + `do_an.ipynb` trong khi code dùng UCI + `do_an_2.ipynb`; thiếu `requirements.txt`; import thừa (`shap`, `fontTools`); không lưu model.

---

## 4. Mức độ ưu tiên sửa (đã thực hiện — xem mục 6 đối chiếu)

| Ưu tiên | Việc | Trạng thái |
|---------|------|-----------|
| P0 | Bỏ cell 111, giữ `category`; split theo thời gian; ANOVA/MI sau split (fit trên train) | ✅ xong trong pipeline `src/` |
| P0 | Sửa bug `evaluate_model` (mask global) và `get_best_model`; split 1 lần dùng chung index | ✅ xong |
| P1 | Xử lý skew target + clip ≥0; loại `atemp`; bỏ feature bình phương trùng Poly; VIF | ✅ xong (VIF + clip; log-target/Poisson thử nghiệm — xem 6.3) |
| P1 | Thêm baseline `cnt` trực tiếp | ✅ xong — 2-model thắng baseline |
| P2 | Cyclic encoding, target-encoding có CV | ✅ cyclic xong |
| P2 | Đồng bộ README + `requirements.txt`, lưu pipeline, dọn import | ✅ xong |

---

## 5. Kết luận đánh giá gốc

- **Điểm mạnh:** EDA công phu, pipeline `ColumnTransformer` + `GridSearchCV` có cấu trúc tốt, có phân tích phần dư và RMSLE.
- **Điểm yếu chí mạng:** leakage, đảo kiểu categorical, bug đánh giá, không xử lý outlier/skew dù đã phát hiện.
- **Khuyến nghị:** không dùng số MAE/R² cũ để kết luận; chạy lại từ train theo thời gian.

---

## 6. Cải tiến đã triển khai (pipeline `src/`, 2026-09-11)

Notebook gốc được tách thành 6 notebook giai đoạn (`src/01_eda.ipynb → 06_evaluate.ipynb` + `run_all.ipynb`), mỗi giai đoạn lưu output riêng vào `datas/01_eda → 06_evaluation`. Toàn bộ P0/P1/P2 ở mục 4 đã áp dụng, cộng thêm các cải tiến hiệu quả mô hình:

### 6.1. Features mới (`03_features`)

- Lag/rolling **chỉ dùng quá khứ** (`shift`): `lag1/lag24`, `roll3/roll24` cho `casual/registered/cnt` (NaN đầu chuỗi điền 0) — tính trên full chuỗi đã sort nên test vẫn dùng đúng đuôi train, không leakage.
- `comfort = temp*(1-hum)`, `is_bad_weather = weathersit>=3`; giữ cyclic sin/cos, rush theo domain, `temp_hum_interaction`; bỏ `atemp`.

### 6.2. Models mới (`05_train`)

- Prep tách 3 nhánh: Linear (Poly+Scale+OneHot) **chỉ** cho Ridge/Ridge-log; GLM (Scale+OneHot, không Poly) cho Poisson/Tweedie; Tree (passthrough+Ordinal) cho Tree/RF/HistGB.
- Từ 3 lên **7 model families**: Ridge, **Ridge-log** (`TransformedTargetRegressor` log1p/expm1), **Poisson**, **Tweedie(1.5)**, DecisionTree, RandomForest, **HistGradientBoosting** (early stopping thật).
- `RandomizedSearchCV` (12 iters) cho RF/HGB để giới hạn chi phí; `GridSearchCV` cho lưới nhỏ; `TimeSeriesSplit(5)`, scoring `neg_MAE`.

### 6.3. Đánh giá mới (`06_evaluate`)

- Hiệu chỉnh count: `clip(>=0)` + làm tròn nguyên; thêm MAE; fix bug mask global.
- So sánh `sum(casual+registered)` vs baseline `cnt`; **metrics theo phân khúc** (`segments_test.csv`): peak, weekend, bad_weather, night, workingday.

### 6.4. Kết quả test (theo thời gian, 3.476 giờ cuối, `datas/06_evaluation/metrics.csv`)

| target | R² | RMSE | RMSLE | MAPE | MAE | model tốt nhất (CV) |
|---|---|---|---|---|---|---|
| casual | 0.942 | 13.44 | 0.47 | 43.4% | 8.17 | HistGB (CV MAE 8.66) |
| registered | 0.951 | 41.78 | 0.29 | 25.0% | 25.98 | HistGB (CV MAE 27.96) |
| cnt trực tiếp | 0.949 | 49.95 | 0.58 | 56.2% | 33.70 | PolyLinear (CV MAE 33.41) |
| **sum(casual+registered)** | **0.956** | **46.15** | **0.29** | **24.1%** | **28.94** | — |

Phân khúc (sum model): peak R² 0.923, weekend 0.956, workingday 0.956, **bad_weather 0.859** (n=238, MAPE 45.5%), night R² 0.839 (MAPE cao do giá trị nhỏ).

### 6.5. So với notebook gốc và remote

- Notebook gốc (số cũ, còn leakage): casual R² 0.983/RMSE 6.34 (lạc quan giả), registered 0.901/46.66, tổng 0.927/48.07. Pipeline mới thắng ở `registered` và tổng hợp trên **mọi** metric; số `casual` cũ đẹp bất thường chính là dấu hiệu leakage.
- Remote `origin/main` (An, RF-only): tổng R² 0.902/RMSE 68.9 — thua pipeline mới (0.956/46.2) nhờ lag features + HistGB.

### 6.6. Việc còn lại

1. Cải thiện phân khúc thời tiết xấu (R² 0.859) — thêm feature mưa/gió, thử sample-weight cho `weathersit>=3`.
2. Điều tra Poisson/Tweedie/Ridge-log CV tệ (R² âm): nghi lag features scale lớn — thử chuẩn hóa robust hoặc loại lag khỏi GLM.
3. Backtest rolling-origin nhiều fold thay vì 1 lát cắt 80/20; kiểm định Diebold-Mariano cho 2-model vs baseline.
4. Thống nhất với remote (`src/` vs `notebooks/`) trước khi merge vào `main`.
