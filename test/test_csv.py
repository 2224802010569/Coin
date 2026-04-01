import os
from datetime import datetime
import pandas as pd

EXPORT_DIR = "test_ans"

def ensure_export_dir():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    return EXPORT_DIR

def csv_pd(name, df: pd.DataFrame):
    if df is None or df.empty:
        print("⚠️ Không có dữ liệu để xuất CSV.")
        return None

    ensure_export_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}.csv" or f"export_{timestamp}.csv"
    path = os.path.join(EXPORT_DIR, filename)

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ Đã xuất {len(df)} dòng ra file: {path}")
    return path
