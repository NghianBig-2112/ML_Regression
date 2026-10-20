import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. THIẾT LẬP ĐƯỜNG DẪN DỮ LIỆU
DATA_PATH = "/kaggle/input/datasets/rohankayan/years-of-experience-and-salary-dataset/Salary_Data.csv"

print(f"--> Đang đọc dữ liệu từ : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.info())