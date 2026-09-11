# Dự đoán nhu cầu thuê xe đạp theo giờ

Dự án phân tích dữ liệu và xây dựng mô hình hồi quy dự đoán số lượt thuê xe đạp theo giờ của hệ thống **Capital Bikeshare**. Thay vì dự đoán trực tiếp `cnt`, dự án huấn luyện hai mô hình riêng cho:

- `casual`: khách thuê vãng lai;
- `registered`: khách hàng đã đăng ký.

Nhu cầu tổng được suy ra bằng `casual + registered`. Cách tiếp cận này giúp phản ánh rõ hơn khác biệt trong hành vi thuê xe của hai nhóm người dùng.

## Mục tiêu

- Khám phá ảnh hưởng của thời gian, mùa, ngày làm việc và thời tiết đến nhu cầu thuê xe.
- Tạo đặc trưng cho giờ cao điểm và hoạt động giải trí cuối tuần.
- So sánh các mô hình hồi quy bằng cross-validation cho chuỗi thời gian.
- Đánh giá mô hình trên tập kiểm tra với R², RMSE, RMSLE và MAPE.

## Dữ liệu

Dữ liệu sử dụng là [Bike Sharing Dataset của UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), ghi nhận theo giờ trong giai đoạn 2011–2012.

- File gốc: `datas/hour.csv`
- Quy mô: 17.379 bản ghi, 17 cột
- Biến mục tiêu: `casual`, `registered`; `cnt = casual + registered`
- Các biến đầu vào chính: `yr`, `mnth`, `hr`, `weekday`, `holiday`, `workingday`, `season`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`

Các biến thời tiết đã được chuẩn hoá theo tài liệu của UCI.

## Quy trình

```text
hour.csv
  ↓
01 — Khám phá và làm sạch dữ liệu
  ↓
02 — Feature engineering, chia train/test theo thời gian
  ↓
03 — Huấn luyện, GridSearchCV và TimeSeriesSplit
  ↓
04 — Đánh giá, chẩn đoán residual và suy luận
```

Hai đặc trưng hành vi được bổ sung:

- `is_rush_hour`: 07:00–09:00 và 17:00–19:00 trong ngày làm việc.
- `is_weekend_leisure`: 10:00–16:00 trong ngày không làm việc.

Dữ liệu được chia tuần tự theo thời gian, với 80% đầu dùng để huấn luyện và 20% cuối dùng để kiểm tra, nhằm tránh rò rỉ thông tin từ tương lai. Trong quá trình tinh chỉnh, `TimeSeriesSplit` dùng 5 folds và khoảng cách 24 giờ giữa train/validation.

## Kết quả hiện tại

Kết quả dưới đây là trên tập kiểm tra gồm 3.476 bản ghi, được ghi nhận trong `notebooks/04_model_evaluation_and_inference.ipynb`.

| Mục tiêu | R² | RMSE | RMSLE | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Casual | 0,8748 | 19,81 | 0,5563 | 52,95% |
| Registered | 0,8957 | 60,79 | 0,4055 | 37,46% |
| Tổng (`casual + registered`) | 0,9023 | 68,91 | 0,3973 | 36,46% |

Ở kết quả cross-validation hiện tại, XGBoost là mô hình tốt nhất cho `casual`, còn LightGBM là mô hình tốt nhất cho `registered` (chọn theo MSE trung bình). Các mô hình đã huấn luyện được lưu bằng `joblib` tại `notebooks/models/`.

## Cấu trúc thư mục

```text
.
├── datas/
│   ├── hour.csv                         # Dữ liệu gốc
│   ├── X_train_*.csv, X_test_*.csv      # Đặc trưng sau xử lý
│   └── y_train_*.csv, y_test_*.csv      # Nhãn train/test
├── notebooks/
│   ├── 01_data_exploration_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_and_cv.ipynb
│   ├── 04_model_evaluation_and_inference.ipynb
│   └── models/
│       ├── best_casual_rf.joblib
│       └── best_registered_rf.joblib
├── do_an_2.ipynb                        # Notebook tổng hợp
└── README.md
```

> Tên tệp `.joblib` có hậu tố `_rf` được giữ lại để tương thích với notebook đánh giá; chúng lưu mô hình tốt nhất của lần huấn luyện hiện tại, không nhất thiết luôn là Random Forest.

## Cài đặt

Yêu cầu Python 3.9 trở lên. Tạo môi trường ảo và cài đặt các thư viện cần thiết:

```bash
python -m venv .venv
```

Trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn lightgbm xgboost joblib
```

## Cách chạy

Từ thư mục gốc của dự án, khởi động Jupyter:

```bash
jupyter notebook
```

Sau đó mở và chạy notebook theo thứ tự:

1. `notebooks/01_data_exploration_and_cleaning.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_model_training_and_cv.ipynb`
4. `notebooks/04_model_evaluation_and_inference.ipynb`

Các notebook sử dụng đường dẫn tương đối. Hãy chạy chúng với thư mục làm việc là `notebooks/` để các đường dẫn `../datas/...` và `models/...` hoạt động đúng.

## Mô hình và lưu ý

Pipeline so sánh Linear/Poisson Regression (baseline), Decision Tree, Random Forest, LightGBM và XGBoost. Không sử dụng `casual`, `registered` hoặc `cnt` làm đặc trưng đầu vào, nhằm tránh rò rỉ dữ liệu.

Chạy lại notebook 02 sẽ tạo lại các tệp train/test trong `datas/`; chạy lại notebook 03 sẽ ghi đè các mô hình `.joblib` trong `notebooks/models/`.
