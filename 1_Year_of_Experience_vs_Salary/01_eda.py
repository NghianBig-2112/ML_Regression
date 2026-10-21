import argparse

import os
import pandas as pd
import matplotlib.pyplot as plt

# NHẬP THAM SỐ TỪ TERMINAL
p = argparse.ArgumentParser()
p.add_argument("--data_dir", type=str, default=None)
p.add_argument("--output_dir", type=str, default="/kaggle/working")
args = p.parse_args()

# 1. ĐỌC DỮ LIỆU
DATA_PATH = args.data_dir
data = pd.read_csv(DATA_PATH)

# 2. kIỂM TRA DỮ LIỆU
print("5 dòng đầu tiên")
print(data.head())

print("Thông tin chi tiết:")
print(data.info())

# 3. Kiểm tra NULL
data_null = data.isna().sum()
print("Danh sách các feature bị NULL")
print(data_null)

# 4. KKIỂM TRA TRÙNG LẶP
data_duplicated = data.duplicated().sum()
print("Danh sách các feature bị trùng lặp")
print(data_duplicated)

# 5. THỐNG KÊ MÔ TẢ
print("Thống kê mô tả")
print(data.describe())

# 6. ĐỘ TƯƠNG QUAN (correlation)
data_corr = data.corr()
print("Hệ số tương quan (Correlation)")
print(data_corr)

# 7. Vẽ biểu đồ
output_path = args.output_dir
plt.figure(figsize=(8, 5))
plt.scatter(
    data["YearsExperience"],
    data["Salary"],
    color="blue",
    alpha=0.7,
    edgcolors="black"
)
plt.title(
    "Mối quan hệ giữa Số năm kinh nghiệm và Mức lương",
    fontsize=24,
    fontweight="bold"
)
plt.xlabel("Số năm kinh nghiệm (Years of Experience)", fontsize=12)
plt.ylabel("Mức lương (Salary)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
# LƯU BIỂU ĐỒ
scatter_path = os.path.join(output_path, "eda_scatter_plot.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
plt.close()