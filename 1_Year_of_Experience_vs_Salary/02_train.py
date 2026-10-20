import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# 6. LƯU MÔ HÌNH RA FILE (.joblib)
MODEL_PATH = "/kaggle/working"
model_save_path = os.path.join(MODEL_PATH, "linear_regression_salary.")
# Dùng joblib để nén và lưu model
joblib.dump(model, model_save_path)
print(f"\n--> Đã đóng gói và lưu mô hình tại: {model_save_path}")

# 7. TIẾN HÀNH DỰ ĐOÁN TRÊN TẬP TEST
y_pred = model.predict(X_test)

# Tạo bảng so sánh Thực tế (Actual) vs Dự đoán (Predicted)
comparison_df = pd.DataFrame(
    {
        "Số năm KN": X_test["YearsExperience"].values,
        "Lương thực tế": y_test.values,
        "Lương dự đoán": y_pred.round(2),
        "Chênh lệch (Sai số)": (y_test.values - y_pred).round(2),
    }
)
print(comparison_df.to_string(index=False))

# 8. Loss function
# 1. MAE (Mean Absolute Error): Trung bình độ lệch tuyệt đối
mae = mean_absolute_error(y_test, y_pred)
# 2. MSE (Mean Squared Error) và RMSE (Root MSE):
#    RMSE có cùng đơn vị với tiền tệ (USD), phạt nặng các dự đoán bị lệch quá lớn
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# 3. R2 Score (Hệ số xác định):
#    Dao động từ (-vô cùng đến 1.0). Càng sát 1.0 chứng tỏ mô hình dự đoán càng xuất sắc
r2 = r2_score(y_test, y_pred)

print(f"1. MAE  (Sai số tuyệt đối trung bình): {mae:,.2f} USD")
print(f"2. RMSE (Căn sai số bình phương TB) : {rmse:,.2f} USD")
print(f"3. R² Score (Hệ số xác định)        : {r2:.4f} (tức ~ {r2 * 100:.2f}%)")

# ==========================================
# 9. TRỰC QUAN HÓA KẾT QUẢ ĐÁNH GIÁ (PLOTTING)
# ==========================================
plt.figure(figsize=(9, 5))
# Vẽ các điểm dữ liệu thực tế của tập Test (Màu xanh dương)
plt.scatter(
    X_test,
    y_test,
    color="blue",
    s=70,
    label="Dữ liệu thực tế (Test Actual)",
    edgecolors="black",
)
# Vẽ đường thẳng dự đoán của Linear Regression (Đường màu đỏ)
plt.plot(
    X_test,
    y_pred,
    color="red",
    linewidth=2,
    label=f"Đường hồi quy dự đoán (R² = {r2:.2f})",
)
plt.title("Đánh giá mô hình: Thực tế vs Đường hồi quy dự đoán", fontsize=14)
plt.xlabel("Số năm kinh nghiệm", fontsize=12)
plt.ylabel("Mức lương (USD)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
# Lưu ảnh biểu đồ đánh giá
eval_plot_path = os.path.join(output_dir, "03_evaluation_plot.png")
plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
print(f"\n--> Đã lưu biểu đồ đánh giá tại: {eval_plot_path}")
plt.close()