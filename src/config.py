"""Cấu hình đường dẫn dùng chung cho toàn pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "datas" / "hour.csv"

EDA_DIR = ROOT / "datas" / "01_eda"
CLEANED_DIR = ROOT / "datas" / "02_cleaned"
FEATURES_DIR = ROOT / "datas" / "03_features"
SPLIT_DIR = ROOT / "datas" / "04_split"
MODELS_DIR = ROOT / "datas" / "05_models"
EVAL_DIR = ROOT / "datas" / "06_evaluation"

CAT_COLS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "time_period"]
TARGETS = ["casual", "registered", "cnt"]
TEST_SIZE = 0.2  # 20% tương lai làm test (split theo thời gian, không shuffle)
RANDOM_STATE = 42
