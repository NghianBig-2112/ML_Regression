import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. THIẾT LẬP ĐƯỜNG DẪN DỮ LIỆU
DATA_PATH = "/kaggle/input/datasets/rohankayan/years-of-experience-and-salary-dataset/Salary_Data.csv"

print(f"--> Đang đọc dữ liệu từ : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# 2. KIỂM TRA TỔNG QUAN DỮ LIỆU
print("\n" + "=" * 40)
print("5 DÒNG ĐẦU TIÊN CỦA DỮ LIỆU")
print(df.head())

print("\n" + "=" * 40)
print("THÔNG TIN CHI TIẾT")
print(df.info())