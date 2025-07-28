"""
File chạy chính cho pipeline phân loại email spam.
"""
import numpy as np
import time
import signal
import pandas as pd
import logging
import os
import argparse
from spam_classifier import SpamClassifierPipeline
from knn_classifier import KNNClassifier
from config import SpamClassifierConfig
from email_handler import EmailHandler
from evaluator import ModelEvaluator
from tfidf_classifier import TFIDFClassifier  # Thêm import

# Thiết lập logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'spam_classifier.log')),
        logging.StreamHandler()
    ]
)
logger = logging.info

# Biến kiểm soát thoát an toàn
running = True


def signal_handler(sig: int, frame: str) -> None:
    """
    Xử lý tín hiệu Ctrl+C để thoát chương trình an toàn.

    Args:
        sig: ID của tín hiệu.
        frame: Frame hiện tại.
    """
    global running
    running = False
    logger("Nhận tín hiệu Ctrl+C. Đang dừng chương trình an toàn...")


def prepare_evaluation_data(evaluator: ModelEvaluator, config: SpamClassifierConfig) -> tuple:
    """
    Tải và chuẩn bị dữ liệu cho đánh giá mô hình.

    Args:
        evaluator: Đối tượng ModelEvaluator để tải dữ liệu và tạo embedding.
        config: Đối tượng cấu hình chứa thông tin dataset và mô hình.

    Returns:
        Tuple chứa (test_embeddings, test_metadata) cho đánh giá.
    """
    # Tải dữ liệu
    logger("Đang tải dữ liệu để đánh giá...")
    messages, labels = evaluator.data_loader.load_data()

    # Tạo embedding
    logger(f"Đang tạo embedding cho {len(messages)} tin nhắn...")
    embeddings = evaluator.embedding_generator.generate_embeddings(messages)

    # Chia dữ liệu thành tập train/test
    logger("Đang chia dữ liệu thành tập train và test...")
    train_idx, test_idx, _, _ = evaluator.data_loader.split_data(messages, labels)

    # Chuẩn bị embedding cho tập test
    test_embeddings = embeddings[test_idx]

    # Tạo metadata cho tập test
    logger("Đang tạo metadata cho tập test...")
    encoded_labels = evaluator.data_loader.label_encoder.transform(labels)
    metadata = evaluator.data_loader.create_metadata(messages, labels, encoded_labels)
    test_metadata = [metadata[i] for i in test_idx]

    return test_embeddings, test_metadata


def main():
    """
    Hàm chính để chạy pipeline phân loại email spam.
    """
    parser = argparse.ArgumentParser(description="Chạy pipeline phân loại email spam.")
    parser.add_argument('--regenerate', action='store_true',
                        help='Tái tạo embedding (mặc định: False)')
    parser.add_argument('--run-email-classifier', action='store_true',
                        help='Chạy chế độ phân loại email qua Gmail API (mặc định: False)')
    parser.add_argument('--merge-emails', action='store_true',
                        help='Gộp email từ thư mục inbox/spam vào dataset (mặc định: False)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Chạy đánh giá mô hình với biểu đồ trực quan (mặc định: False)')
    parser.add_argument('--k-values', type=str,
                        help='Danh sách giá trị k cho đánh giá, phân tách bằng dấu phẩy (ví dụ: "1,3,5")')
    parser.add_argument('--classifier', type=str, default='knn', choices=['knn', 'tfidf'],
                        help='Chọn bộ phân loại: knn (mặc định) hoặc tfidf')
    args = parser.parse_args()

    try:
        # Khởi tạo cấu hình
        config = SpamClassifierConfig()
        config.regenerate_embeddings = args.regenerate
        if args.k_values:
            config.k_values = [int(k) for k in args.k_values.split(',')]
        logger(f"Tái tạo embedding: {config.regenerate_embeddings}")
        logger(f"Giá trị k: {config.k_values}")

        # Tạo pipeline
        logger("Đang khởi tạo pipeline...")
        pipeline = SpamClassifierPipeline(config, classifier_type=args.classifier)

        # Gộp email nếu được yêu cầu
        if args.merge_emails:
            logger("Đang gộp email từ thư mục inbox/spam vào dataset...")
            pipeline.data_loader.merge_emails_to_dataset()

            # Kiểm tra tính nhất quán giữa dataset và cache embedding
            dataset_path = config.dataset_path
            embeddings_file = os.path.join('cache', 'embeddings',
                                          f"embeddings_{config.model_name.replace('/', '_')}.npy")
            if os.path.exists(dataset_path) and os.path.exists(embeddings_file):
                df = pd.read_csv(dataset_path)
                dataset_count = len(df)
                embeddings = np.load(embeddings_file)
                cache_count = embeddings.shape[0]
                if cache_count != dataset_count and not args.regenerate:
                    logger(
                        f"CẢNH BÁO: Số dòng trong dataset ({dataset_count}) không khớp với cache embedding ({cache_count}). "
                        "Chạy lại với --regenerate để cập nhật embedding."
                    )
                    return
                elif cache_count != dataset_count and args.regenerate:
                    logger("Phát hiện số dòng không khớp. Đang tái tạo embedding...")

        # Huấn luyện mô hình (chỉ một lần cho pipeline chính)
        logger("Đang bắt đầu huấn luyện mô hình...")
        pipeline.train()

        # Đánh giá mô hình nếu được yêu cầu
        if args.evaluate:
            logger("Đang bắt đầu đánh giá mô hình...")
            evaluator = ModelEvaluator(config)
            test_embeddings, test_metadata = prepare_evaluation_data(evaluator, config)
            
            # Init TF-IDF pipeline riêng (train chỉ một lần)
            tfidf_pipeline = SpamClassifierPipeline(config, classifier_type='tfidf')
            tfidf_pipeline.train()
            
            evaluator.evaluate_accuracy(test_embeddings, test_metadata, pipeline.classifier, tfidf_pipeline.classifier, config.k_values)
            return

        # Chạy phân loại email nếu được yêu cầu
        if args.run_email_classifier:
            logger("Đang khởi động chế độ phân loại email qua Gmail API ở chế độ nền...")
            handler = EmailHandler(pipeline, config)
            last_page_token = None

            # Đăng ký trình xử lý tín hiệu Ctrl+C
            signal.signal(signal.SIGINT, signal_handler)

            while running:
                try:
                    # Lấy danh sách email mới
                    results = handler.service.users().messages().list(
                        userId='me',
                        q='is:unread',
                        maxResults=10,
                        includeSpamTrash=True,
                        pageToken=last_page_token
                    ).execute()
                    messages = results.get('messages', [])

                    if messages:
                        logger(f"Phát hiện {len(messages)} email mới. Đang xử lý...")
                        handler.process_emails(max_results=10)
                        last_page_token = results.get('nextPageToken')
                    else:
                        logger("Không có email mới. Chờ 30 giây...")

                    time.sleep(30)
                except Exception as e:
                    logger(f"Lỗi khi xử lý email: {str(e)}")
                    time.sleep(60)

            logger("Chương trình đã dừng an toàn.")
            return

        # Kiểm tra với các ví dụ mẫu
        logger("Đang kiểm tra pipeline với các ví dụ mẫu...")
        test_examples = [
            "I am actually thinking a way of doing something useful",
            "FREE!! Click here to win $1000 NOW! Limited time offer!"
        ]
        for i, example in enumerate(test_examples, 1):
            logger(f"Ví dụ {i}: {example}")
            result = pipeline.predict(example, k=3)
            logger(f"Dự đoán cho ví dụ {i}: {result['prediction']}")

    except Exception as e:
        logger(f"Lỗi trong quá trình chạy chính: {str(e)}")
        raise


if __name__ == "__main__":
    main()