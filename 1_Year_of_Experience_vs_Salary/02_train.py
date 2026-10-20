import pandas as pd
from sklearn.model_selection import train_test_split

# 1. ĐỌC DỮ LIỆU
DATA_PATH = "/kaggle/input/datasets/rohankayan/years-of-experience-and-salary-dataset/Salary_Data.csv"
print(f"--> Đang đọc dữ liệu từ : {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# 2. CHIA DỮ LIỆU THÀNH TRAIN - TEST DATASET
TARGET = "Salary"
X = data.drop(TARGET, axis=1)
y = data[TARGET]
print(f"Kích thước của X (features): {X.shape}")
print(f"Kích thước của y (target): {y.shape}")
