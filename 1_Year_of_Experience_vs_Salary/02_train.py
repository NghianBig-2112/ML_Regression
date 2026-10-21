import argparse

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

p = argparse.ArgumentParser()
p.add_argument("--data_dir", type=str, default=None)
p.add_argument("--output_dir", type=str, default="/kaggle/working")
args = p.parse_args()

# 1. ĐỌC DỮ LIỆU
print("-- Đang đọc dữ liệu")
DATA_PATH = args.data_dir
data = pd.read_csv(DATA_PATH)

# 2. SPLIT DATA 
print("-- Đang chia dữ liệu")
TARGET = "Salary"
X = data.drop(TARGET, axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"-- Số lượng X train:{len(X_train)}")
print(f"-- Số lượng X test:{len(X_test)}")

# 3. KHỎI TẠO MÔ HÌNH REGRESSION
model = LinearRegression()

# 4. HUẤN LUYỆN MÔ HÌNH
model.fit(X_train, y_train)

print(f"-- Hệ số góc w = {model.coef_[0]}")
print(f"-- Bias b = {model.interccept_}")

# 5. LƯU MÔ HÌNH
OUTPUT_PATH = args.output_dir
model_save = os.path.join(OUTPUT_PATH, "linear_regression_salary.joblib")
joblib.dump(model, model_save)
print(f"-- Model đã được save tại: {model_save}")

# 8. Dữ đoán trên tập X_test
y_pred = model.predict(X_test)

print("Bảng so sánh Actual vs Predicted")
comparison_df = df.DataFrame(
    {
        "Year of Experience": X_test["YearExperience"].values,
        "Actual Salary": y_test.values,
        "Predicted Salary": y_pred.round(2),
        "Chênh lệch": (y_test.values - y_pred).round(2)
    }
)
print(comparison_df.to_string(index=False))

# 9. Loss function
# MAE - Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# MSE - Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# RMSE - Root MSE: có cùng đơn vị với tiền tệ (USD), phạt nặng các dự đoán bị lệch quá lớn
rmse = np.sqrt(mse)

# R2 Score: Dao động từ (-vô cùng đến 1.0). Càng sát 1.0 chứng tỏ mô hình dự đoán càng xuất sắc
r2 = r2_score(y_test, y_pred)

print(f"-- Mean absolute error: {mae}")
print(f"-- Mean squared error: {mse}")
print(f"-- Root Mean squared error: {rmse}")
print(f"-- R2 score: {r2}")

# 10. Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.scatter(
    data["YearsExperience"],
    data["Salary"],
    color="blue",
    alpha=0.7,
    edgecolors="black"
)
plt.plot(X, y_pred, color="red", label="Đường hồi quy")
plt.title("Mối quan hệ giữa Số năm kinh nghiệm và Mức lương", fontsize=24)

plt.xlabel("Số năm kinh nghiệm (Years of Experience)", fontsize=12)
plt.ylabel("Mức lương (Salary)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

eval_plot_path = os.path.join(output_dir, "line_linear_plot.png")
plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
plt.close()