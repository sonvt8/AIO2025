"""
Cấu hình cho hệ thống phân loại email spam/ham.
"""
from dataclasses import dataclass
from typing import List
import os


@dataclass
class SpamClassifierConfig:
    """Cấu hình cho spam classifier."""
    
    # Cài đặt mô hình
    model_name: str = 'intfloat/multilingual-e5-base'
    max_length: int = 512
    batch_size: int = 32
    
    # Cài đặt huấn luyện
    test_size: float = 0.1
    random_state: int = 42
    
    # Cài đặt KNN
    default_k: int = 3
    k_values: List[int] = None
    
    # Đường dẫn
    dataset_path: str = './dataset/2cls_spam_text_cls.csv'
    output_dir: str = './cache/output'
    output_file: str = os.path.join(output_dir, 'error_analysis.json')
    
    # Flag để yêu cầu có hoặc không tạo lại embedding
    regenerate_embeddings: bool = False
    
    # Gmail API settings
    credentials_path: str = './cache/input/credentials.json'
    token_path: str = './cache/input/token.json'
    
    # Đường dẫn thư mục local để lưu email
    inbox_local_dir: str = './inbox'
    spam_local_dir: str = './spam'
    
    def __post_init__(self):
        """Khởi tạo các giá trị mặc định sau khi tạo object."""
        if self.k_values is None:
            self.k_values = [1, 3, 5]
        # Tạo các thư mục nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.credentials_path), exist_ok=True)
        os.makedirs(self.inbox_local_dir, exist_ok=True)
        os.makedirs(self.spam_local_dir, exist_ok=True)