"""Chạy toàn pipeline theo thứ tự. Dừng khi gặp lỗi."""
import runpy
from pathlib import Path

SRC = Path(__file__).resolve().parent
for f in ["01_eda.py", "02_clean.py", "03_features.py", "04_split.py", "05_train.py", "06_evaluate.py"]:
    print(f"\n{'='*60}\n>>> {f}\n{'='*60}")
    runpy.run_path(str(SRC / f), run_name="__main__")
print("\nPipeline HOÀN TẤT. Xem kết quả trong datas/01_eda .. datas/06_evaluation")
