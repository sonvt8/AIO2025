import streamlit as st

# Hàm tính Precision, Recall, F1-score
def calculate_f1_components(tp, fp, fn):
    # Kiểm tra kiểu dữ liệu: phải là int
    if not (isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int)):
        raise ValueError("Các giá trị TP, FP, FN phải là số nguyên (integer). Vui lòng kiểm tra lại!")

    # Kiểm tra giá trị: phải >= 0
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError("Các giá trị TP, FP, FN không được nhỏ hơn 0. Vui lòng nhập lại!")

    # Kiểm tra trường hợp chia cho 0
    errors = []
    if tp + fp == 0:
        errors.append("Tổng TP + FP = 0 (Precision không xác định).")
    if tp + fn == 0:
        errors.append("Tổng TP + FN = 0 (Recall không xác định).")
    if errors:
        error_msg = "Không thể tính toán vì:\n" + "\n".join([f"- {error}" for error in errors]) + "\nVui lòng nhập lại!"
        raise ValueError(error_msg)

    # Tính Precision, Recall, F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        f1_score = 0.0  # Trường hợp đặc biệt khi cả Precision và Recall đều là 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

# Tiêu đề ứng dụng
st.title("Tính Precision, Recall và F1-score")

# Mô tả
st.write("Nhập các giá trị TP (True Positive), FP (False Positive), FN (False Negative) để tính các chỉ số đánh giá mô hình phân loại.")

# Form nhập liệu
st.subheader("Nhập dữ liệu")
col1, col2, col3 = st.columns(3)
with col1:
    tp = st.number_input("True Positive (TP):", min_value=0, step=1, format="%d")
with col2:
    fp = st.number_input("False Positive (FP):", min_value=0, step=1, format="%d")
with col3:
    fn = st.number_input("False Negative (FN):", min_value=0, step=1, format="%d")

# Nút tính toán
if st.button("Tính toán"):
    try:
        # Tính toán
        precision, recall, f1_score = calculate_f1_components(tp, fp, fn)

        # Hiển thị thông báo thành công
        st.success("Tính toán thành công! Kết quả được hiển thị bên dưới.")

        # Hiển thị kết quả dạng bảng
        st.subheader("Kết quả")
        result = {
            "Chỉ số": ["Precision", "Recall", "F1-score"],
            "Giá trị": [f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}"]
        }
        st.table(result)

    except ValueError as e:
        # Hiển thị lỗi dạng thông báo thân thiện với gạch đầu dòng
        st.error(f"⚠️ {str(e)}")
    except Exception as e:
        # Thông báo lỗi chung
        st.error(f"⚠️ Đã xảy ra lỗi không xác định: {str(e)}. Vui lòng thử lại!")

# Hiển thị công thức
st.subheader("Công thức")
st.markdown("""
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)  
- **F1-score** = 2 * (Precision * Recall) / (Precision + Recall)
""")