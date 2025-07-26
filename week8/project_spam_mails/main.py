"""
File chạy chính cho spam classifier.
"""
import numpy as np
import time
import signal
import pandas as pd
import logging
import os
import argparse
from spam_classifier import SpamClassifierPipeline
from config import SpamClassifierConfig
from email_handler import EmailHandler

# Thiết lập logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'spam_classifier.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Biến kiểm soát thoát
running = True

def signal_handler(sig, frame):
    """Xử lý tín hiệu Ctrl+C để thoát an toàn."""
    global running
    running = False
    logger.info("Nhận tín hiệu thoát. Đang dừng chương trình một cách an toàn...")

def main():
    """Hàm chính để chạy spam classifier."""
    parser = argparse.ArgumentParser(description="Run spam classifier pipeline.")
    parser.add_argument('--regenerate', action='store_true', default=False,
                        help='Set to regenerate embeddings (default: False)')
    parser.add_argument('--run-email-classifier', action='store_true', default=False,
                        help='Run email classifier mode with Gmail API (default: False)')
    parser.add_argument('--merge-emails', action='store_true', default=False,
                        help='Merge emails from inbox/spam folders into dataset (default: False)')
    args = parser.parse_args()
    
    try:
        # Khởi tạo cấu hình
        config = SpamClassifierConfig()
        config.regenerate_embeddings = args.regenerate
        logger.info(f"Regenerate embeddings: {config.regenerate_embeddings}")
        
        # Tạo pipeline
        pipeline = SpamClassifierPipeline(config)
        logger.info("Khởi tạo pipeline thành công")
        
        # Gộp email từ thư mục inbox/spam nếu có flag
        if args.merge_emails:
            logger.info("Gộp email từ thư mục inbox/spam vào dataset")
            pipeline.data_loader.merge_emails_to_dataset()
            
            # Kiểm tra số dòng dataset so với embeddings cache
            dataset_path = config.dataset_path
            embeddings_file = os.path.join('cache', 'embeddings', f"embeddings_{config.model_name.replace('/', '_')}.npy")
            if os.path.exists(dataset_path) and os.path.exists(embeddings_file):
                try:
                    df = pd.read_csv(dataset_path)
                    dataset_count = len(df)
                    embeddings = np.load(embeddings_file)
                    cache_count = embeddings.shape[0]
                    if cache_count != dataset_count and not args.regenerate:
                        logger.warning(
                            f"CẢNH BÁO: Số dòng trong dataset ({dataset_count}) không khớp với embeddings cache ({cache_count}). "
                            "Vui lòng chạy lại với --regenerate để cập nhật embeddings trước khi tiếp tục."
                        )
                        return  # Dừng chương trình tại đây
                    elif cache_count != dataset_count and args.regenerate:
                        logger.info(f"Số dòng không khớp, sẽ regenerate embeddings...")
                except Exception as e:
                    logger.error(f"Lỗi khi kiểm tra dataset hoặc embeddings cache: {str(e)}")
                    raise
        
        # Huấn luyện mô hình (chỉ chạy một lần)
        logger.info("Bắt đầu quá trình huấn luyện")
        pipeline.train()
        
        if args.run_email_classifier:
            # Mode classify email qua Gmail API chạy nền
            logger.info("Bắt đầu mode classify email qua Gmail API ở chế độ nền")
            handler = EmailHandler(pipeline, config)
            last_page_token = None
            
            # Đăng ký handler cho tín hiệu Ctrl+C
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
                        logger.info(f"Phát hiện {len(messages)} email mới. Bắt đầu xử lý...")
                        handler.process_emails(max_results=10)
                        last_page_token = results.get('nextPageToken')
                    else:
                        logger.info("Không có email mới. Chờ 30 giây...")
                    
                    time.sleep(30)  # Delay 30 giây trước khi kiểm tra lại
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý email: {str(e)}")
                    time.sleep(60)  # Delay lâu hơn nếu lỗi để tránh spam API
            
            logger.info("Chương trình đã dừng an toàn.")
        else:
            # Test với các ví dụ khác nhau
            test_examples = [
                "I am actually thinking a way of doing something useful",
                "FREE!! Click here to win $1000 NOW! Limited time offer!"
            ]
            logger.info("Đang test pipeline với các ví dụ khác nhau")
            for i, example in enumerate(test_examples, 1):
                logger.info(f"Ví dụ {i}: {example}")
                result = pipeline.predict(example, k=3)
                logger.info(f"Dự đoán cho ví dụ {i}: {result['prediction']}")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chạy main: {str(e)}")
        raise

if __name__ == "__main__":
    main()