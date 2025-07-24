"""
File chạy chính cho spam classifier.
"""
import numpy as np
from spam_classifier import SpamClassifierPipeline
from config import SpamClassifierConfig


def main():
    """Hàm chính để chạy spam classifier."""
    # Khởi tạo cấu hình
    config = SpamClassifierConfig()
    
    # Tạo pipeline
    pipeline = SpamClassifierPipeline(config)
    
    # Huấn luyện mô hình
    pipeline.train()
    
    # Test với các ví dụ khác nhau
    test_examples = [
        "I am actually thinking a way of doing something useful",
        "FREE!! Click here to win $1000 NOW! Limited time offer!"
    ]
    
    print("\nĐang test pipeline với các ví dụ khác nhau:")
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n***Ví dụ {i}:")
        result = pipeline.predict(example, k=3)
        print()


if __name__ == "__main__":
    main()
