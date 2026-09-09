# Dự đoán số lượt thuê xe theo giờ

## Giới thiệu

Đồ án phân tích và dự đoán nhu cầu thuê xe đạp công cộng theo giờ, dùng **UCI Bike Sharing Dataset** (Capital Bikeshare 2011–2012).
Mục tiêu là khám phá mối quan hệ giữa số lượt thuê xe với thời gian, mùa, ngày lễ/ngày làm việc và thời tiết, rồi xây dựng mô hình hồi quy dự đoán `cnt`.

> File `datas/SeoulBikeData.csv` là dữ liệu Seoul cũ, hiện không dùng. Dataset chính là `datas/hour.csv`.

## Nội dung hiện có

Notebook `do_an_2.ipynb` (152 cells) thực hiện:

- Đọc `hour.csv`, ép `dteday→datetime`, giữ `season/yr/mnth/hr/holiday/weekday/workingday/weathersit/time_period` ở dạng `category`.
- Kiểm tra null, phân tích skew/kurtosis, phát hiện outlier IQR vs Z-score (chỉ báo cáo, chưa loại bỏ).
- EDA `casual` vs `registered` theo giờ/tháng/ngày/mùa, workingday/holiday, thời tiết; ma trận Spearman.
- Feature engineering không leakage: `is_weekend`, cyclic `hr/mnth/weekday (sin/cos)`, rush-hour theo domain (`morning/evening/midday`), `time_period`, `temp_hum_interaction`; bỏ `atemp` để giảm đa cộng tuyến (giữ `temp`).
- Chọn đặc trưng ANOVA + Mutual Information **chỉ trên train** (split theo thời gian 80/20, quá khứ → train).
- Train Ridge / DecisionTree / RandomForest bằng `GridSearchCV + TimeSeriesSplit`, so sánh 2 hướng: `casual+registered` cộng lại vs baseline `cnt` trực tiếp.
- Đánh giá test R²/RMSE/RMSLE/MAPE (đã fix bug mask global, sort bảng, clip dự báo ≥ 0), kiểm tra VIF, phân tích phần dư, lưu model `joblib`.

Chi tiết các lỗi đã sửa xem `danh_gia_project.md`.

## Dữ liệu

`datas/hour.csv` (~17.379 dòng × 17 cột, UCI):

- **Target:** `casual`, `registered`, `cnt = casual + registered`
- **Thời gian:** `dteday`, `yr`, `mnth`, `hr`, `weekday`
- **Ngày:** `holiday`, `workingday`, `season`
- **Thời tiết/môi trường:** `weathersit`, `temp`, `hum`, `windspeed` (`atemp` đã loại để tránh cộng tuyến với `temp`)

## Cài đặt

Yêu cầu Python 3.9 trở lên:

```bash
pip install -r requirements.txt
```

## Cách chạy

1. Mở terminal tại thư mục gốc của dự án.
2. Khởi động Jupyter:

   ```bash
   jupyter notebook
   ```

3. Mở `do_an_2.ipynb`.
4. Chạy các cell theo thứ tự từ trên xuống dưới (đã sắp xếp: split thời gian → selection trên train → train → đánh giá).

Notebook đọc dữ liệu bằng đường dẫn tương đối `./datas/hour.csv`, cần mở notebook từ thư mục gốc.

## Cấu trúc thư mục

```text
.
├── do_an_2.ipynb
├── datas/
│   ├── hour.csv            # dataset chính (UCI)
│   └── SeoulBikeData.csv   # dữ liệu cũ, không dùng
├── requirements.txt
├── danh_gia_project.md     # báo cáo đánh giá chi tiết
└── README.md
```

Sau khi chạy train sẽ sinh thêm `best_casual.pkl`, `best_registered.pkl`, `best_cnt.pkl` (đã gitignore nếu cần).

## Một số nhận xét

- Nhu cầu thuê xe thay đổi rõ rệt theo giờ, ngày làm việc/nghỉ, mùa và thời tiết.
- `registered` chiếm tỷ trọng lớn, ổn định theo giờ đi làm; `casual` nhạy với cuối tuần, trưa và thời tiết đẹp.
- Nhiệt độ tương quan dương, độ ẩm tương quan âm với lượt thuê; `temp/atemp` cộng tuyến mạnh nên chỉ giữ `temp`.
- Target lệch phải mạnh: dùng thêm RMSLE, clip ≥ 0; nếu cần tốt hơn hãy thử `log1p`-target hoặc Poisson/Tweedie.

## Hướng phát triển

- Thử `TransformedTargetRegressor(log1p)` / `PoissonRegressor` cho count data.
- Thêm lag/rolling theo giờ-ngày từ `dteday`.
- Thử Gradient Boosting (XGB/LGBM) với `RandomizedSearchCV`.
- Xây dựng demo nhập điều kiện thực tế để dự đoán `cnt`.
