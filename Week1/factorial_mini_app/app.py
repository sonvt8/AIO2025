import streamlit as st
from factorial import factorial

# Tiêu đề của ứng dụng
st.title("Tính Giai Thừa")

# Mô tả ngắn
st.write("Nhập một số nguyên không âm để tính giai thừa.")

# Ô nhập liệu
n = st.number_input("Nhập số nguyên n:", min_value=0, step=1, format="%d")

# Nút tính toán
if st.button("Tính Giai Thừa"):
    try:
        # Tính giai thừa sử dụng module factorial
        result = factorial(n)
        st.success(f"Giai thừa của {n} là: {result}")
    except ValueError as e:
        st.error(f"Lỗi: {e}")
    except OverflowError:
        st.error("Số quá lớn để tính giai thừa!")
    except RecursionError:
        st.error("Số quá lớn, vượt quá giới hạn đệ quy!")

#streamlit run app.py