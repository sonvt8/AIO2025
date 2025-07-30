# app.py

import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE

from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from evaluator import ModelEvaluator
from spam_classifier import SpamClassifierPipeline

# --- Cấu hình trang và CSS tùy chỉnh ---
st.set_page_config(page_title="Bảng điều khiển Spam Mail", layout="centered")
st.markdown("""
<style>
/* Chỉnh theme tối toàn cục */
html, body, [class*="css"] {
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
/* Hộp thư */
.folder-box {
    background-color: #1f2937;
    border-radius: 8px;
    padding: 0.2rem 0.5rem; /* Giảm padding-top để sát mép */
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
.folder-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #3b82f6;
    display: flex;
    align-items: center;
    margin-top: 0; /* Sát mép trên */
}
.folder-count {
    background-color: #ef4444;
    color: white;
    border-radius: 50%;
    padding: 0.2rem 0.6rem;
    margin-left: 0.5rem;
    font-size: 0.9rem;
}
/* Style trực tiếp cho button trong list */
.folder-box .stButton > button {
    background-color: #374151;
    margin: 0.3rem 0;
    width: 100%;
    text-align: left;
}
.folder-box .stButton > button:hover {
    background-color: #4b5563;
    transform: translateY(-2px);
}
/* Nội dung mails */
.content-container {
    background-color: #1f2937;
    border-radius: 8px;
    padding: 1rem;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: flex-start;
    color: #d1d5db;
    overflow-y: auto;
    max-width: 100%;        /* hạn chế tràn chiều ngang */
    word-wrap: break-word;  /* cho phép ngắt dòng nếu quá dài */
    white-space: pre-wrap;  /* giữ định dạng xuống dòng nếu có */
}
.placeholder {
    color: #9ca3af;
    font-style: italic;
    text-align: center;
    flex-grow: 1; /* Giãn để lấp container khi rỗng */
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# --- Tải và cache pipeline theo type và tái sử dụng ---
@st.cache_resource
def load_knn_pipeline():
    cfg = SpamClassifierConfig()
    pipeline = SpamClassifierPipeline(cfg, classifier_type="knn")
    pipeline.train()
    return pipeline

@st.cache_resource
def load_tfidf_pipeline():
    cfg = SpamClassifierConfig()
    pipeline = SpamClassifierPipeline(cfg, classifier_type="tfidf")
    pipeline.train()
    return pipeline


# --- Tải dữ liệu mẫu vào session_state ---
@st.cache_data
def load_sample_data(path: str) -> pd.DataFrame:
    """
    Đọc file CSV chứa dữ liệu email (Category, Message).
    """
    return pd.read_csv(path)


@st.cache_data
def get_embeddings(msgs: list) -> np.ndarray:
    """
    Sinh hoặc load embeddings cho danh sách messages.
    """
    eg = EmbeddingGenerator(SpamClassifierConfig())
    return eg.generate_embeddings(msgs)


@st.cache_data
def get_embeddings_cached(messages: list) -> np.ndarray:
    """Cache embeddings generation."""
    cfg = SpamClassifierConfig()
    eg = EmbeddingGenerator(cfg)
    return eg.generate_embeddings(messages)


@st.cache_data
def compute_tsne_cached(sub_emb: np.ndarray) -> np.ndarray:
    """Cache t-SNE computation."""
    return TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(sub_emb)


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
    embeddings = get_embeddings_cached(messages)  # Cache Emb

    n_samples = min(1000, embeddings.shape[0])
    idx = np.random.choice(embeddings.shape[0], size=n_samples, replace=False)
    sub_emb = embeddings[idx]
    sub_lbl = [df["Category"].iloc[i] for i in idx]

    with st.spinner("Đang tính toán t-SNE…"):
        proj = compute_tsne_cached(sub_emb)  # Cache t-SNE

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

    # Khởi tạo DataLoader, Embedding và Pipeline
    cfg = SpamClassifierConfig()
    loader = DataLoader(cfg)
    messages, labels = loader.load_data()
    emb = EmbeddingGenerator(cfg).generate_embeddings(messages)
    train_idx, test_idx, _, _ = loader.split_data(messages, labels)
    test_emb = emb[test_idx]
    test_meta = []
    encoded = loader.label_encoder.transform(labels)
    metadata = loader.create_metadata(messages, labels, encoded)
    for i in test_idx:
        test_meta.append(metadata[i])

    # Train hai mô hình KNN và TF-IDF
    pipe_knn = SpamClassifierPipeline(cfg, classifier_type="knn")
    pipe_knn.train()
    pipe_tfidf = SpamClassifierPipeline(cfg, classifier_type="tfidf")
    pipe_tfidf.train()

    evaluator = ModelEvaluator(cfg)

    # Chạy evaluate_accuracy — giả sử return combined_results, knn_errors, combined_cms
    with st.spinner("Đang đánh giá mô hình, xin chờ…"):
        combined_results, knn_errors, combined_cms = evaluator.evaluate_accuracy(
            test_embeddings=test_emb,
            test_metadata=test_meta,
            knn_classifier=pipe_knn.classifier,
            tfidf_classifier=pipe_tfidf.classifier,
            k_values=cfg.k_values
        )
    st.success("✅ Đánh giá đã hoàn tất!")

    # Lấy data từ results và cms
    knn_results = combined_results['knn']
    tfidf_results = combined_results['tfidf']
    best_k = combined_results['best_k']
    knn_best = knn_results[best_k]
    knn_cms = combined_cms['knn']
    tfidf_cm = combined_cms['tfidf']

    # 1. Thông báo giá trị K tốt nhất và Accuracy tương ứng
    best_acc = knn_best["accuracy"]
    st.info(f"🔎 K tốt nhất: **k = {best_k}**, Accuracy = **{best_acc:.4f}**")

    # 2. Lineplot KNN metrics
    st.subheader("So sánh chỉ số KNN theo k")
    fig_metrics = evaluator.plot_knn_metrics(knn_results, cfg.k_values)
    st.pyplot(fig_metrics)

    # 3. Heatmaps KNN per k (dùng columns để ngang hàng)
    st.subheader("Confusion Matrix KNN theo k")
    cols = st.columns(len(cfg.k_values))
    for idx, k in enumerate(cfg.k_values):
        with cols[idx]:
            fig_cm = evaluator.plot_knn_confusion(knn_cms[k], k)
            st.pyplot(fig_cm)

    # 4. Barplot phân bố labels
    st.subheader("Phân bố nhãn")
    fig_dist = evaluator.plot_label_distribution(labels)
    st.pyplot(fig_dist)

    # 5. Grouped bar so sánh TF-IDF vs best KNN
    st.subheader("So sánh TF-IDF vs Best KNN")
    fig_comp = evaluator.plot_comparison(knn_best, tfidf_results, best_k)
    st.pyplot(fig_comp)

    # 6. Heatmap TF-IDF
    st.subheader("Confusion Matrix TF-IDF")
    fig_tfidf_cm = evaluator.plot_tfidf_confusion(tfidf_cm)
    st.pyplot(fig_tfidf_cm)


# --- Trang Lấy Thư (đang phát triển) ---
elif st.session_state.page == "✉️ Lấy Thư":
    st.header("✉️ Lấy Thư từ Gmail")

    # Load pipelines riêng biệt
    knn_pipeline = load_knn_pipeline()
    tfidf_pipeline = load_tfidf_pipeline()

    # Selectbox chọn classifier (mặc định KNN để tương thích cũ)
    if 'selected_classifier' not in st.session_state:
        st.session_state['selected_classifier'] = "KNN (FAISS - Semantic Similarity)"
    selected_classifier = st.selectbox(
        "Chọn phương pháp phân loại:",
        ["KNN (FAISS - Semantic Similarity)", "TF-IDF (Term Frequency)"],
        index=0 if st.session_state['selected_classifier'] == "KNN (FAISS - Semantic Similarity)" else 1,
        key='classifier_select'
    )
    st.session_state['selected_classifier'] = selected_classifier  # Lưu trạng thái để giữ khi rerun

    # (Tùy chọn) Thêm expander giải thích ngắn gọn cho UX tốt hơn
    with st.expander("Thông tin về phương pháp phân loại"):
        st.markdown("""
        - **KNN (FAISS - Semantic Similarity)**: Phân loại dựa trên độ tương đồng ngữ nghĩa sử dụng embeddings và vector database (chậm hơn nhưng chính xác cao với ngữ cảnh phức tạp).
        - **TF-IDF (Term Frequency)**: Phân loại dựa trên tần suất từ khóa (nhanh hơn, phù hợp với dữ liệu văn bản đơn giản).
        """)

    if st.button("🔄 Fetch Emails Mới", use_container_width=True):
        with st.spinner(f"Đang fetch và phân loại emails bằng {selected_classifier}..."):
            # Chọn pipeline dựa trên lựa chọn
            classifier_pipeline = knn_pipeline if "KNN" in selected_classifier else tfidf_pipeline

            # --- GIẢ LẬP DỮ LIỆU ---
            raw_emails = [
                {"id": "new_1", "body": "Hello, this is a friendly reminder about our meeting tomorrow."},
                {"id": "new_2", "body": "URGENT: Your account has been compromised! Click here to secure it NOW!"},
                {"id": "new_3", "body": "Check out our latest newsletter for exciting updates."},
                {"id": "new_4", "body": "EXCLUSIVE OFFER just for you! Win a free iPhone 15, limited time only."},
            ]
            # --- KẾT THÚC GIẢ LẬP ---

            # 2. Phân loại và lưu vào session_state
            st.session_state['inbox_emails'] = []
            st.session_state['spam_emails'] = []

            for email in raw_emails:
                # Lấy dictionary kết quả từ pipeline
                result = classifier_pipeline.predict(email['body'])
                # Lấy giá trị dự đoán từ key 'prediction'
                prediction = result['prediction'] 
                
                if prediction == 'ham':
                    st.session_state['inbox_emails'].append(email)
                else:
                    st.session_state['spam_emails'].append(email)
            
            st.success(f"Đã lấy và phân loại {len(raw_emails)} emails bằng {selected_classifier}!")
            # Reset email đang được chọn để tránh hiển thị email cũ
            st.session_state['selected_email'] = None

    # Khởi tạo session_state nếu chưa có (giữ nguyên)
    if 'inbox_emails' not in st.session_state:
        st.session_state['inbox_emails'] = []
    if 'spam_emails' not in st.session_state:
        st.session_state['spam_emails'] = []
    if 'selected_email' not in st.session_state:
        st.session_state['selected_email'] = None

    col_left, col_middle, col_right = st.columns([1, 3, 1])

    # --- Cột Inbox (Bên trái) --- (giữ nguyên)
    with col_left:
        st.markdown('<div class="folder-box">', unsafe_allow_html=True)
        inbox_count = len(st.session_state.inbox_emails)
        st.markdown(f'<div class="folder-title">📥 Inbox <span class="folder-count">{inbox_count}</span></div>', unsafe_allow_html=True)
        for email in st.session_state.inbox_emails:
            if st.button(email['id'], key=f"inbox_{email['id']}", use_container_width=True):
                st.session_state['selected_email'] = {"body": email['body'], "from": "Inbox"}
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Cột Content (Giữa) --- (giữ nguyên)
    with col_middle:
        content_html = ""
        if st.session_state['selected_email'] is None:
            content_html = """
            <div class="content-container">
                <div class="placeholder">
                    Chọn một email từ Inbox hoặc Spam để xem nội dung 📧
                </div>
            </div>
            """
        else:
            selected = st.session_state['selected_email']
            from html import escape
            body_content = escape(selected.get('body', 'N/A')).replace('\n', '<br>')
            from_folder = escape(selected.get('from', 'N/A'))
            content_html = f"""
            <div class="content-container">
                <p><b>From Folder:</b> {from_folder}</p>
                <hr>
                <p>{body_content}</p>
            </div>
            """
        st.markdown(content_html, unsafe_allow_html=True)

    # --- Cột Spam (Bên phải) --- (giữ nguyên)
    with col_right:
        st.markdown('<div class="folder-box">', unsafe_allow_html=True)
        spam_count = len(st.session_state.spam_emails)
        st.markdown(f'<div class="folder-title">🗑️ Spam <span class="folder-count">{spam_count}</span></div>', unsafe_allow_html=True)
        for email in st.session_state.spam_emails:
            if st.button(email['id'], key=f"spam_{email['id']}", use_container_width=True):
                st.session_state['selected_email'] = {"body": email['body'], "from": "Spam"}
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# --- Footer ---
st.markdown("<footer>Được xây dựng với Streamlit | Vận hành bởi pipeline AI của bạn.</footer>", unsafe_allow_html=True)