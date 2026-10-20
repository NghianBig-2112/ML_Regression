import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. THIẾT LẬP ĐƯỜNG DẪN DỮ LIỆU
DATA_PATH = "/kaggle/input/datasets/rohankayan/years-of-experience-and-salary-dataset/Salary_Data.csv"

print(f"--> Đang đọc dữ liệu từ : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# 2. KIỂM TRA TỔNG QUAN DỮ LIỆU
print("5 DÒNG ĐẦU TIÊN CỦA DỮ LIỆU")
print("\n" + "=" * 40)
print(df.head())

print("THÔNG TIN CHI TIẾT")
print("\n" + "=" * 40)
print(df.info())

print(f"\n Kích thước tập dữ liệu (row, column): {df.shape}")

# 3. KIỂM TRA DỮ LIỆU BỊ THIẾU VÀ TRÙNG LẶP
print("KIỂM TRA DỮ LIỆU THIẾU (NULL / MISSING VALUES):")
print("=" * 40)
# Đếm số lượng ô bị trống ở từng cột
missing_values = df.isna().sum()
print(missing_values)

print("KIỂM TRA DỮ LIỆU TRÙNG LẶP (DUPLICATES):")
print("=" * 40)
duplicates = df.duplicated().sum()
print(f"Số dòng bị trùng lặp: {duplicates}")

# 4. THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)
print("BẢNG THỐNG KÊ MÔ TẢ:")
print("=" * 40)
# Cung cấp: min, max, mean (trung bình), std (độ lệch chuẩn), các mốc phân vị 25%, 50%, 75%
print(df.describe())

# 5. ĐỘ TƯƠNG QUAN (CORRELATION)
print("MA TRẬN HỆ SỐ TƯƠNG QUAN (PEARSON CORRELATION):")
print("=" * 40)
# Giá trị tiến gần 1.0 tức là kinh nghiệm càng tăng thì lương càng tăng mạnh (tương quan thuận rất cao)
correlation = df.corr()
print(correlation)

# 6. TRỰC QUAN HÓA DỮ LIỆU (VISUALIZATION)
# --- Biểu đồ 1: Scatter Plot (Quan trọng nhất cho Linear Regression) ---
output_dir = "/kaggle/working/image_eda"
plt.figure(figsize=(8, 5))
plt.scatter(
    df["YearsExperience"],
    df["Salary"],
    color="blue",
    alpha=0.7,
    edgecolors="black",
)
plt.title(
    "Mối quan hệ giữa Số năm kinh nghiệm và Mức lương",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Số năm kinh nghiệm (Years of Experience)", fontsize=12)
plt.ylabel("Mức lương (Salary)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
# Lưu biểu đồ thành file ảnh để xem lại trên Kaggle
scatter_path = os.path.join(output_dir, "01_eda_scatter_plot.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
print(f"\n--> Đã lưu biểu đồ phân tán tại: {scatter_path}")
plt.close()