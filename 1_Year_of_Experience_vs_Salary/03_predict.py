import os
import joblib
import pandas as pd

# ==========================================
# 1. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN
# ==========================================
output_dir = "/kaggle/working" if os.path.exists("/kaggle") else "."
model_path = os.path.join(output_dir, "linear_regression_salary.joblib")

if not os.path.exists(model_path):
    print(f"Lỗi: Không tìm thấy file mô hình tại '{model_path}'!")
    print("Vui lòng chạy file '02_train.py' trước để tạo ra file mô hình.")
    exit()

print(f"--> Đang nạp mô hình từ: {model_path}")
model = joblib.load(model_path)
print("--> Nạp mô hình thành công!\n")


# ==========================================
# 2. HÀM DỰ ĐOÁN MỨC LƯƠNG CHO 1 NGƯỜI
# ==========================================
def predict_salary(years_of_experience):
    """Hàm nhận vào số năm kinh nghiệm và trả về mức lương dự đoán.

    Lưu ý: Scikit-learn yêu cầu dữ liệu truyền vào hàm predict() phải có đúng
    định dạng 2 chiều (DataFrame có tên cột giống lúc huấn luyện:
    'YearsExperience') để tránh cảnh báo UserWarning.
    """
    input_data = pd.DataFrame({"YearsExperience": [years_of_experience]})
    predicted_salary = model.predict(input_data)[0]
    return predicted_salary


# ==========================================
# 3. KỊCH BẢN 1: DỰ ĐOÁN CHO MỘT DANH SÁCH ỨNG VIÊN (BATCH PREDICTION)
# ==========================================
print("=" * 55)
print("KỊCH BẢN 1: DỰ ĐOÁN HÀNG LOẠT THEO DANH SÁCH CÓ SẴN")
print("=" * 55)

# Danh sách số năm kinh nghiệm của các ứng viên mới cần định giá lương
candidates_experience = [0.5, 2.5, 4.0, 7.5, 10.0]

for exp in candidates_experience:
    salary = predict_salary(exp)
    print(f"Kinh nghiệm: {exp:4.1f} năm  -->  Lương ước tính: ${salary:10,.2f}")

# ==========================================
# 4. KỊCH BẢN 2: CHO PHÉP NGƯỜI DÙNG TỰ NHẬP TỪ BÀN PHÍM (TÙY CHỌN)
# ==========================================
print("\n" + "=" * 55)
print("KỊCH BẢN 2: TỰ NHẬP SỐ NĂM KINH NGHIỆM")
print("=" * 55)

try:
    user_input = input("Nhập số năm kinh nghiệm bạn muốn kiểm tra (ví dụ 3.5): ")
    if user_input.strip() != "":
        exp_value = float(user_input)
        if exp_value < 0:
            print("Số năm kinh nghiệm không thể âm!")
        else:
            estimated_salary = predict_salary(exp_value)
            print(f"\n=> KẾT QUẢ DỰ BÁO:")
            print(f"   Ứng viên có {exp_value} năm kinh nghiệm")
            print(f"   Mức lương thị trường ước tính: ${estimated_salary:,.2f}")
except ValueError:
    print("Dữ liệu nhập vào không hợp lệ (phải là số thực hoặc số nguyên).")
except Exception as e:
    # Bỏ qua nếu chạy ở môi trường không có bàn phím tương tác
    pass

print("\n[HOÀN TẤT] File 04_predict.py đã chạy xong!")