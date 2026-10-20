import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegrsssion

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Số lượng X_train: {len(X_train)}")
print(f"Số lượng X_test: {len(X_test)}")

# 3. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH
# Model linear regression
model = LinearRegression()

# Hàm fit() chính là lúc mô hình 'học' từ dữ liệu Train:
model.fit(X_train, y_train)
print("--> Huấn luyện thành công!")

print(f"Hệ số góc w = {model.coef_[0]}")
print(f"Bias b = {model.intercept_}")