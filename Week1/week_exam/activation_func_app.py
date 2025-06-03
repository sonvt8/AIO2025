import streamlit as st
import math

# Hàm tính Sigmoid và đạo hàm
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    derivative = sig * (1 - sig)
    return sig, derivative

# Hàm tính ReLU và đạo hàm
def relu(x):
    value = max(0, x)
    derivative = 1 if x > 0 else 0
    return value, derivative

# Hàm tính ELU và đạo hàm (alpha = 0.01)
def elu(x, alpha=0.01):
    if x > 0:
        value = x
        derivative = 1
    else:
        value = alpha * (math.exp(x) - 1)
        derivative = value + alpha
    return value, derivative

# Tiêu đề ứng dụng
st.title("Tính Giá Trị và Đạo Hàm Hàm Kích Hoạt")

# Mô tả
st.write("Nhập giá trị x và chọn hàm kích hoạt để tính giá trị và đạo hàm.")

# Form nhập liệu
st.subheader("Nhập dữ liệu")
col1, col2 = st.columns(2)
with col1:
    x = st.number_input("Nhập giá trị x:", value=1.5, step=0.1, format="%.1f")
with col2:
    activation = st.selectbox("Chọn hàm kích hoạt:", ["sigmoid", "relu", "elu"])

# Nút tính toán
if st.button("Tính toán"):
    try:
        # Tính giá trị và đạo hàm
        if activation == 'sigmoid':
            value, derivative = sigmoid(x)
        elif activation == 'relu':
            value, derivative = relu(x)
        else:  # elu
            value, derivative = elu(x, alpha=0.01)

        # Hiển thị kết quả dạng bảng
        st.subheader("Kết quả")
        result = {
            "Thông số": [f"Hàm {activation}(x)", f"Đạo hàm {activation}'(x)"],
            "Giá trị": [f"{value:.2f}", f"{derivative:.2f}"]
        }
        st.table(result)

    except Exception as e:
        st.error(f"⚠️ Lỗi không xác định: {str(e)}. Vui lòng thử lại!")

# Hiển thị công thức
st.subheader("Công thức")
if activation == 'sigmoid':
    st.markdown(r"""
    - **Giá trị Sigmoid**: $\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}$  
    - **Đạo hàm Sigmoid**: $\text{sigmoid}'(x) = \text{sigmoid}(x) \cdot (1 - \text{sigmoid}(x))$
    """)
    st.markdown("**Ghi chú**: Đạo hàm của Sigmoid được tính dựa trên giá trị của hàm Sigmoid tại \( x \).")
elif activation == 'relu':
    st.markdown(r"""
    - **Giá trị ReLU**: $\text{relu}(x) = \begin{cases} 0 & \text{nếu } x \leq 0 \\ x & \text{nếu } x > 0 \end{cases}$  
    - **Đạo hàm ReLU**: $\text{relu}'(x) = \begin{cases} 0 & \text{nếu } x \leq 0 \\ 1 & \text{nếu } x > 0 \end{cases}$
    """)
    st.markdown("**Ghi chú**: Đạo hàm của ReLU là không liên tục tại \( x = 0 \), thường được gán giá trị 0 hoặc 1 tùy theo ngữ cảnh.")
else:  # elu
    st.markdown(r"""
    - **Giá trị ELU**: $\text{elu}(x) = \begin{cases} x & \text{nếu } x > 0 \\ \alpha (e^x - 1) & \text{nếu } x \leq 0 \end{cases}$, với $\alpha = 0.01$  
    - **Đạo hàm ELU**: $\text{elu}'(x) = \begin{cases} 1 & \text{nếu } x > 0 \\ \text{elu}(x) + \alpha & \text{nếu } x \leq 0 \end{cases}$
    """)
    st.markdown("**Ghi chú**: Đạo hàm của ELU tại \( x \leq 0 \) phụ thuộc vào giá trị của hàm ELU tại \( x \).")