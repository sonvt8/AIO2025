import os
import streamlit as st
import pandas as pd
from factorial import factorial

def load_users():
    """Đọc danh sách user từ file user.txt"""
    try:
        if os.path.exists("users.xlsx"):
            df = pd.read_excel("users.xlsx", dtype={'pass': str})
            users_dict = dict(zip(df['user'], df['pass']))
            return users_dict
        else:
            st.error("File users.xlsx không tồn tại!")
            return []
    except Exception as e:
        st.error(f"Lỗi khi đọc file user.txt: {e}")
    return []


def greeting_page():
    """Trang chào hỏi cho user không hợp lệ"""
    st.title(f"Xin chào {st.session_state.username}!")
    st.markdown(
        """
        :orange-badge[⚠️ Bạn không có quyền truy cập vào chức năng tính giai thừa.]
        """,
        unsafe_allow_html=True
    )
    if st.button("Quay lại đăng nhập"):
        st.session_state.show_greeting = False
        st.session_state.username = ""
        st.rerun()


def login_page():
    st.title("Đăng nhập")
    # Input username
    username = st.text_input("username:")
    pwd = st.text_input("pass:", type="password")
    if st.button("Đăng nhập"):
        if username and pwd:
            users = load_users()
            user_pass = users.get(username)
            if user_pass is not None and user_pass.strip() == str(pwd).strip():
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                # user không hợp lệ
                st.session_state.show_greeting = True
                st.session_state.username = username
                st.rerun()
        else:
            st.warning("Vui lòng đủ thông tin username và password")

 
def factorial_calculator():
    # Tiêu đề của ứng dụng
    st.title("Tính Giai Thừa")

    # Nút logout ở sidebar
    with st.sidebar:
        st.write(f"👋 Xin chào {st.session_state.username}")
        if st.button("Đăng xuất"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

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
                
def main():
    # Khởi tạo session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'show_greeting' not in st.session_state:
        st.session_state.show_greeting = False
    # Điều hướng trang dựa trên trạng thái đăng nhập
    if st.session_state.logged_in:
        factorial_calculator()
    elif st.session_state.show_greeting:
        greeting_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
