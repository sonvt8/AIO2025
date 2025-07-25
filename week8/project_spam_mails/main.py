"""
File chạy chính cho spam classifier.
"""
import numpy as np
from spam_classifier import SpamClassifierPipeline
from config import SpamClassifierConfig
import logging
import os
import argparse  # Thêm import argparse để parse CLI args

# Thiết lập logging tập trung
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

def main():
    """Hàm chính để chạy spam classifier."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run spam classifier pipeline.")
    parser.add_argument('--regenerate', action='store_true', default=False,
                        help='Set to regenerate embeddings (default: False)')
    args = parser.parse_args()
    
    try:
        # Khởi tạo cấu hình
        config = SpamClassifierConfig()
        # Cập nhật flag từ CLI
        config.regenerate_embeddings = args.regenerate
        logger.info(f"Regenerate embeddings: {config.regenerate_embeddings}")
        
        # Tạo pipeline
        pipeline = SpamClassifierPipeline(config)
        logger.info("Khởi tạo pipeline thành công")
        
        # Huấn luyện mô hình
        logger.info("Bắt đầu quá trình huấn luyện")
        pipeline.train()
        
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