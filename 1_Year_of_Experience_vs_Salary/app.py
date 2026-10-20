import os
from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Nạp model đã huấn luyện
MODEL_PATH = os.path.join("model", "linear_regression_salary.joblib")
model = joblib.load(MODEL_PATH)


# 1. Trang chủ: Hiển thị giao diện HTML
@app.route("/")
def home():
    return render_template("index.html")


# 2. Xử lý dự đoán khi người dùng bấm nút trên HTML
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu năm kinh nghiệm từ form HTML gửi lên
        exp = float(request.form.get("experience"))

        # Dự đoán
        input_data = pd.DataFrame({"YearsExperience": [exp]})
        salary = model.predict(input_data)[0]

        # Trả lại trang HTML kèm kết quả
        return render_template(
            "index.html",
            prediction_text=f"${salary:,.2f} USD",
            years_experience=exp,
        )
    except Exception as e:
        return render_template("index.html", error_text="Vui lòng nhập một số hợp lệ!")


if __name__ == "__main__":
    # Chạy web server tại máy cục bộ: http://127.0.0.1:5000
    app.run(debug=True, port=5000)