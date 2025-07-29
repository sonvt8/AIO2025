# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE

from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from spam_classifier import SpamClassifierPipeline

# --- Cấu hình trang và CSS tùy chỉnh ---
st.set_page_config(page_title="Bảng điều khiển Spam Mail", layout="centered")
st.markdown("""
<style>
/* Chỉnh theme tối toàn cục */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    color: #f0f0f0;
    background-color: #111827;
}
/* Style cho button */
.stButton > button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #2563eb;
    transform: translateY(-2px);
}
/* Tiêu đề lớn và đoạn mô tả */
.main-title {
    font-size: 2.8rem;
    font-weight: bold;
    margin-top: 1rem;
    color: #f9fafb;
    text-align: center;
    text-shadow: 1px 1px 5px #3b82f6;
}
.subtext {
    font-size: 1.1rem;
    line-height: 1.6;
    color: #d1d5db;
    max-width: 800px;
    margin: auto;
}
footer {
    color: #9ca3af;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# --- Tải và cache pipeline để tái sử dụng ---
@st.cache_resource
def load_pipeline():
    """
    Khởi tạo và train pipeline phân loại spam.
    Kết quả được cache để không train lại mỗi lần rerun.
    """
    cfg = SpamClassifierConfig()
    pipeline = SpamClassifierPipeline(cfg)
    pipeline.train()
    return pipeline

pipeline = load_pipeline()


# --- Tải dữ liệu mẫu vào session_state ---
@st.cache_data
def load_sample_data(path: str) -> pd.DataFrame:
    """
    Đọc file CSV chứa dữ liệu email (Category, Message).
    """
    return pd.read_csv(path)

if "df" not in st.session_state:
    st.session_state["df"] = load_sample_data(SpamClassifierConfig().dataset_path)
df = st.session_state["df"]


# --- Quản lý trạng thái trang và nút về trang chủ ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Tổng quan"

if st.session_state.page != "🏠 Tổng quan":
    if st.button("🏠 Trở về Tổng quan"):
        st.session_state.page = "🏠 Tổng quan"
        st.rerun()


# --- Trang Tổng quan (Overview) ---
if st.session_state.page == "🏠 Tổng quan":
    st.markdown('<h1 class="main-title">📧 Bộ phân loại Spam/Ham Mail</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Khám phá và phân loại email với giao diện tương tác!</div>', unsafe_allow_html=True)

    # Thống kê nhanh
    total = len(df)
    spam_cnt = len(df[df["Category"] == "spam"])
    ham_cnt  = len(df[df["Category"] == "ham"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số Email", total)
    c2.metric("Email Spam", spam_cnt, f"{spam_cnt/total*100:.1f}%")
    c3.metric("Email Ham", ham_cnt, f"{ham_cnt/total*100:.1f}%")

    st.markdown("### Tính năng:")

    # Nút chuyển đến từng page
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Phân tích Dữ liệu", use_container_width=True):
            st.session_state.page = "📊 Phân tích Dữ liệu"
            st.rerun()
    with col2:
        if st.button("📈 Đánh giá Bộ phân loại", use_container_width=True):
            st.session_state.page = "📈 Đánh giá Bộ phân loại"
            st.rerun()
    with col3:
        if st.button("✉️ Lấy Thư", use_container_width=True):
            st.session_state.page = "✉️ Lấy Thư"
            st.rerun()


# --- Trang Phân tích Dữ liệu ---
elif st.session_state.page == "📊 Phân tích Dữ liệu":
    st.header("📊 Phân tích Dữ liệu")

    # 1) Biểu đồ cột Spam vs Ham kèm số lượng trên đỉnh cột
    st.subheader("Phân phối Spam vs Ham")
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Nhóm", "Số lượng"]
    fig1 = px.bar(
        counts,
        x="Nhóm",
        y="Số lượng",
        text="Số lượng",
        text_auto=True,
        title="Phân phối Email Spam và Ham",
        color="Nhóm",
        color_discrete_map={"spam":"#ef4444","ham":"#22c55e"},
        labels={"Nhóm":"Loại","Số lượng":"Số lượng Email"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2) t-SNE visualization trên 1.000 mẫu
    st.subheader("Minh họa embedding với t-SNE (1.000 mẫu)")
    messages = df["Message"].tolist()

    @st.cache_data
    def get_embeddings(msgs: list) -> np.ndarray:
        """
        Sinh hoặc load embeddings cho danh sách messages.
        """
        eg = EmbeddingGenerator(SpamClassifierConfig())
        return eg.generate_embeddings(msgs)

    embeddings = get_embeddings(messages)
    n_samples = min(1000, embeddings.shape[0])
    idx = np.random.choice(embeddings.shape[0], size=n_samples, replace=False)
    sub_emb = embeddings[idx]
    sub_lbl = [df["Category"].iloc[i] for i in idx]

    with st.spinner("Đang tính toán t-SNE…"):
        proj = TSNE(n_components=2, init="random", learning_rate="auto")\
               .fit_transform(sub_emb)

    df_vis = pd.DataFrame(proj, columns=["Dim 1","Dim 2"])
    df_vis["Nhóm"] = sub_lbl
    fig2 = px.scatter(
        df_vis,
        x="Dim 1",
        y="Dim 2",
        color="Nhóm",
        title="Phân tán embedding qua t-SNE",
        color_discrete_map={"spam":"#ef4444","ham":"#22c55e"},
        hover_data=["Nhóm"]
    )
    st.plotly_chart(fig2, use_container_width=True)


# --- Trang Đánh giá Bộ phân loại (đang phát triển) ---
elif st.session_state.page == "📈 Đánh giá Bộ phân loại":
    st.header("📈 Đánh giá Bộ phân loại")
    st.info("Chức năng đang được phát triển…")


# --- Trang Lấy Thư (đang phát triển) ---
elif st.session_state.page == "✉️ Lấy Thư":
    st.header("✉️ Lấy Thư từ Gmail")
    st.info("Chức năng đang được phát triển…")


# --- Footer ---
st.markdown("<footer>Được xây dựng với Streamlit | Vận hành bởi pipeline AI của bạn.</footer>", unsafe_allow_html=True)