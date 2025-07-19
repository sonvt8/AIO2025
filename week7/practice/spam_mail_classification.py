"""
Hệ thống phân loại Email Spam

Hệ thống phân loại email thành spam hoặc ham sử dụng
các thuật toán Naive Bayes với xử lý văn bản và trực quan hóa.
"""

import os
from typing import Optional, Tuple, Any
import logging

import gdown
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def DownloadFile(
    file_id: str,
    output_dir: str = "dataset",
    filename: str = "default.csv"
) -> Optional[str]:
    """
    Tải file từ Google Drive bằng ID file.

    Tham số:
        file_id: ID file Google Drive
        output_dir: Thư mục để lưu file
        filename: Tên file đầu ra

    Trả về:
        Đường dẫn đến file đã tải hoặc None nếu thất bại
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        logger.info(f"✅ File đã tồn tại: {output_path}")
        return output_path

    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, output_path, quiet=False)
        logger.info(f"📥 Tải file thành công: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải file: {e}")
        return None


def create_spam_ham_visualization(df: pd.DataFrame) -> None:
    """
    Tạo biểu đồ trực quan hóa phân bố Spam vs Ham.

    Tham số:
        df: DataFrame chứa cột 'Category' với giá trị đã mã hóa
            (0=Ham, 1=Spam)
    """
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8')

    # Bảng màu hiện đại
    colors = ['#FF6B9D', '#4ECDC4']

    # Tạo biểu đồ số lượng
    ax = sns.countplot(
        data=df,
        x='Category',
        palette=colors,
        edgecolor='white',
        linewidth=1.5
    )

    # Tùy chỉnh tiêu đề và nhãn
    plt.title(
        '📧 Phân Bố Spam vs Ham trong Dataset',
        fontsize=20,
        fontweight='bold',
        pad=20,
        color='#2C3E50'
    )
    plt.xlabel(
        'Loại Email',
        fontsize=14,
        fontweight='bold',
        color='#34495E'
    )
    plt.ylabel(
        'Số lượng Email',
        fontsize=14,
        fontweight='bold',
        color='#34495E'
    )

    # Tùy chỉnh nhãn trục x
    plt.xticks(
        [0, 1],
        ['📩 Ham (Bình thường)', '⚠️ Spam (Rác)'],
        fontsize=12,
        fontweight='bold'
    )

    # Thêm thông tin số lượng
    category_counts = df['Category'].value_counts().sort_index()
    max_count = max(category_counts)

    for i, count in enumerate(category_counts):
        # Thêm số lượng
        ax.text(
            i, count + max_count * 0.02,
            f'{count:,}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            color='#2C3E50'
        )

        # Thêm phần trăm
        percentage = (count / len(df)) * 100
        ax.text(
            i, count + max_count * 0.08,
            f'({percentage:.1f}%)',
            ha='center', va='bottom',
            fontsize=10, style='italic',
            color='#7F8C8D'
        )

    # Tùy chỉnh lưới và nền
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#FAFAFA')

    # Định dạng đường viền
    for spine in ax.spines.values():
        spine.set_color('#BDC3C7')
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    y_true: Any,
    y_pred_gauss: Any,
    y_pred_multi: Any,
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Vẽ ma trận nhầm lẫn cho cả hai mô hình cạnh nhau.

    Tham số:
        y_true: Nhãn thực tế
        y_pred_gauss: Dự đoán từ GaussianNB
        y_pred_multi: Dự đoán từ MultinomialNB
        figsize: Kích thước hình
    """
    # Tính ma trận nhầm lẫn
    cm_gauss = confusion_matrix(y_true, y_pred_gauss)
    cm_multi = confusion_matrix(y_true, y_pred_multi)

    # Tạo subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Tham số chung cho heatmap
    heatmap_kwargs = {
        'annot': True,
        'fmt': 'd',
        'cbar_kws': {'label': 'Số lượng'},
        'annot_kws': {'size': 12, 'weight': 'bold'}
    }

    # Heatmap cho GaussianNB
    sns.heatmap(cm_gauss, cmap='Blues', ax=ax1, **heatmap_kwargs)
    ax1.set_title(
        'GaussianNB - Ma trận Nhầm lẫn',
        fontsize=14,
        fontweight='bold'
    )
    ax1.set_xlabel('Dự đoán', fontsize=12)
    ax1.set_ylabel('Thực tế', fontsize=12)
    ax1.set_xticklabels(['Ham', 'Spam'])
    ax1.set_yticklabels(['Ham', 'Spam'])

    # Heatmap cho MultinomialNB
    sns.heatmap(cm_multi, cmap='Reds', ax=ax2, **heatmap_kwargs)
    ax2.set_title(
        'MultinomialNB - Ma trận Nhầm lẫn',
        fontsize=14,
        fontweight='bold'
    )
    ax2.set_xlabel('Dự đoán', fontsize=12)
    ax2.set_ylabel('Thực tế', fontsize=12)
    ax2.set_xticklabels(['Ham', 'Spam'])
    ax2.set_yticklabels(['Ham', 'Spam'])

    plt.tight_layout()
    plt.show()


def preprocess_text(text: str) -> str:
    """
    Xử lý trước văn bản bằng cách chuyển thành chữ thường, loại bỏ dấu câu,
    từ dừng và áp dụng stemming.

    Tham số:
        text: Văn bản đầu vào cần xử lý

    Trả về:
        Văn bản đã xử lý dưới dạng chuỗi
    """
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Chuyển thành chữ thường và loại bỏ dấu câu
    text = text.lower().translate(str.maketrans('', '', string.punctuation))

    # Tách từ, loại bỏ từ dừng và áp dụng stemming
    return ' '.join([
        ps.stem(word) for word in text.split()
        if word not in stop_words
    ])


def predict(
    text: str,
    model: Any,
    vectorizer: CountVectorizer,
    dense: bool = False
) -> int:
    """
    Dự đoán văn bản là spam hay ham.

    Tham số:
        text: Văn bản đầu vào để phân loại
        model: Mô hình đã huấn luyện (GaussianNB hoặc MultinomialNB)
        vectorizer: CountVectorizer đã fit
        dense: Có chuyển sang mảng dày không (cho GaussianNB)

    Trả về:
        Dự đoán (0 cho ham, 1 cho spam)
    """
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])

    if dense:
        features = features.toarray()

    prediction = model.predict(features)
    return prediction[0]


def print_section_header(title: str, width: int = 80) -> None:
    """In tiêu đề phần với định dạng."""
    print("=" * width)
    print(f"🔍 {title}")
    print("=" * width)


def print_dataset_info(df: pd.DataFrame) -> None:
    """In thông tin toàn diện về dataset."""
    print_section_header("THÔNG TIN DATASET")

    print("📊 Thống kê cơ bản về Dataset:")
    print(f"   • Tổng số email: {len(df):,}")
    print(f"   • Các cột: {list(df.columns)}")
    print(f"   • Sử dụng bộ nhớ: \
        {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    print("\n📋 Dữ liệu mẫu (5 dòng đầu):")
    print(df.head().to_string())

    print("\n📈 Mô tả Dataset:")
    print(df.describe())


def print_data_analysis(df: pd.DataFrame) -> None:
    """In phân tích dữ liệu chi tiết."""
    print_section_header("PHÂN TÍCH DỮ LIỆU")

    ham_count = (df['Category'] == 0).sum()
    spam_count = (df['Category'] == 1).sum()
    total = len(df)

    print("📊 Phân tích Phân bố:")
    print(f"   • Tổng số email: {total:,}")
    print(f"   • Email Ham: {ham_count:,} ({ham_count/total*100:.2f}%)")
    print(f"   • Email Spam: {spam_count:,} ({spam_count/total*100:.2f}%)")
    print(f"   • Tỷ lệ Ham:Spam: {ham_count/spam_count:.2f}:1")


def print_model_results(
    model_name: str,
    val_accuracy: float,
    test_accuracy: float,
    y_val: Any,
    y_val_pred: Any,
    y_test: Any,
    y_test_pred: Any
) -> None:
    """In kết quả đánh giá mô hình toàn diện."""
    print_section_header(f"KẾT QUẢ MÔ HÌNH {model_name.upper()}")

    print("🎯 Điểm độ chính xác:")
    print(f"   • Độ chính xác Validation: {val_accuracy:.4f} \
          ({val_accuracy*100:.2f}%)")
    print(f"   • Độ chính xác Test: {test_accuracy:.4f} \
          ({test_accuracy*100:.2f}%)")

    print("\n📊 Báo cáo Phân loại Chi tiết (Validation):")
    print(classification_report(y_val, y_val_pred))

    print("\n📊 Báo cáo Phân loại Chi tiết (Test):")
    print(classification_report(y_test, y_test_pred))


def main() -> None:
    """Hàm thực thi chính."""
    print_section_header("HỆ THỐNG PHÂN LOẠI EMAIL SPAM", 80)

    # Tải dataset
    logger.info("📥 Đang tải dataset...")
    file_path = DownloadFile(
        file_id="1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R",
        output_dir="dataset",
        filename="mails.csv"
    )

    if file_path is None:
        logger.error("❌ Không thể tải file. Dừng chương trình.")
        return

    # Tải và khám phá dữ liệu
    logger.info("📂 Truy xuất dữ liệu trong dataset...")
    mails_df = pd.read_csv(file_path)

    print_dataset_info(mails_df)

    # Mã hóa nhãn
    logger.info("🔄 Đang mã hóa nhãn...")
    mails_df['Category'] = mails_df['Category'].map({'ham': 0, 'spam': 1})

    print_data_analysis(mails_df)

    # Tạo trực quan hóa
    logger.info("📊 Đang tạo trực quan hóa...")
    # create_spam_ham_visualization(mails_df)

    # Xử lý trước văn bản
    logger.info("🔧 Đang xử lý trước dữ liệu văn bản...")
    mails_df['processed_message'] = [
        preprocess_text(msg) for msg in mails_df['Message']
    ]

    # Vector hóa văn bản
    logger.info("🔤 Đang vector hóa văn bản...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(mails_df['processed_message'])
    y = mails_df['Category'].values

    print_section_header("KỸ THUẬT ĐẶC TRƯNG")
    print(f"📏 Kích thước Ma trận Đặc trưng: {X.shape}")
    print(f"📏 Kích thước Nhãn: {y.shape}")
    print(f"📏 Kích thước Từ vựng: {len(vectorizer.vocabulary_):,}")

    # Chia dữ liệu
    logger.info("✂️ Đang chia dataset...")
    VAL_SIZE, TEST_SIZE, SEED = 0.2, 0.125, 0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED
    )

    print_section_header("CHIA DỮ LIỆU")
    print(f"📊 Tập Huấn luyện: {X_train.shape[0]:,} mẫu")
    print(f"📊 Tập Validation: {X_val.shape[0]:,} mẫu")
    print(f"📊 Tập Test: {X_test.shape[0]:,} mẫu")

    # Chuyển ma trận thưa sang dày cho GaussianNB
    logger.info("🔄 Đang chuyển ma trận cho GaussianNB...")
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    X_test_dense = X_test.toarray()

    # Huấn luyện GaussianNB
    logger.info("🤖 Đang huấn luyện mô hình GaussianNB...")
    model_gauss = GaussianNB()
    model_gauss.fit(X_train_dense, y_train)

    # Huấn luyện MultinomialNB
    logger.info("🤖 Đang huấn luyện mô hình MultinomialNB...")
    model_multi = MultinomialNB()
    model_multi.fit(X_train, y_train)

    # Đánh giá mô hình
    logger.info("📊 Đang đánh giá mô hình...")

    # Đánh giá GaussianNB
    y_val_pred_gauss = model_gauss.predict(X_val_dense)
    y_test_pred_gauss = model_gauss.predict(X_test_dense)
    val_acc_gauss = accuracy_score(y_val, y_val_pred_gauss)
    test_acc_gauss = accuracy_score(y_test, y_test_pred_gauss)

    print_model_results(
        "GaussianNB", val_acc_gauss, test_acc_gauss,
        y_val, y_val_pred_gauss, y_test, y_test_pred_gauss
    )

    # Đánh giá MultinomialNB
    y_val_pred_multi = model_multi.predict(X_val)
    y_test_pred_multi = model_multi.predict(X_test)
    val_acc_multi = accuracy_score(y_val, y_val_pred_multi)
    test_acc_multi = accuracy_score(y_test, y_test_pred_multi)

    print_model_results(
        "MultinomialNB", val_acc_multi, test_acc_multi,
        y_val, y_val_pred_multi, y_test, y_test_pred_multi
    )

    # Vẽ ma trận nhầm lẫn
    logger.info("📊 Đang tạo ma trận nhầm lẫn...")
    plot_confusion_matrices(y_val, y_val_pred_gauss, y_val_pred_multi)

    # Kiểm thử dự đoán
    print_section_header("KIỂM THỬ DỰ ĐOÁN")
    test_input = 'I am actually thinking a way of doing something useful'
    label_map = {0: 'ham', 1: 'spam'}

    logger.info("🔮 Đang kiểm thử dự đoán...")

    prediction_gauss = predict(test_input, model_gauss, vectorizer, dense=True)
    prediction_multi = predict(test_input, model_multi, vectorizer)

    print(f"📧 Email kiểm thử: '{test_input}'")
    print(f"🤖 Dự đoán GaussianNB: {label_map[prediction_gauss]} \
        (mã: {prediction_gauss})")
    print(f"🤖 Dự đoán MultinomialNB: {label_map[prediction_multi]} \
        (mã: {prediction_multi})")

    # Tóm tắt so sánh mô hình
    print_section_header("TÓM TẮT SO SÁNH MÔ HÌNH")
    print("🏆 So sánh Hiệu suất:")
    print(f"   • GaussianNB    - Val: {val_acc_gauss:.4f},  \
        Test: {test_acc_gauss:.4f}")
    print(f"   • MultinomialNB - Val: {val_acc_multi:.4f},  \
        Test: {test_acc_multi:.4f}")

    models = {True: "GaussianNB", False: "MultinomialNB"}
    best_model = models[test_acc_gauss > test_acc_multi]
    print(f"🥇 Mô hình Hiệu suất Tốt nhất: {best_model}")

    logger.info("✅ Chương trình hoàn thành thành công!")


if __name__ == "__main__":
    main()
