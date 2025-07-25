"""
Module xử lý và tải dữ liệu cho spam classification.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import logging
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK packages (chỉ run nếu chưa có, quiet để không in thừa)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

logger = logging.getLogger(__name__)

class DataLoader:
    """Class để tải và xử lý dữ liệu."""
    
    def __init__(self, config):
        """
        Khởi tạo DataLoader.
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config
        self.label_encoder = LabelEncoder()
        logger.info("Đã khởi tạo DataLoader")
    
    def preprocess_text(self, text: str) -> str:
        """
        Xử lý trước văn bản: remove URLs/emails/numbers, lowercase, remove punctuation,
        remove stop words, và lemmatize.

        Args:
            text: Văn bản đầu vào cần xử lý

        Returns:
            Văn bản đã xử lý dưới dạng chuỗi
        """
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # Remove URLs, emails, numbers
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
        text = re.sub(r'\S+@\S+', '', text)  # Emails
        text = re.sub(r'\d+', '', text)  # Numbers

        # Lowercase and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))

        # Tokenize, remove stop words, lemmatize
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    
    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Tải dữ liệu từ file CSV và áp dụng preprocess.
        
        Returns:
            Tuple chứa danh sách tin nhắn đã preprocess và nhãn
        
        Raises:
            FileNotFoundError: Nếu file CSV không tồn tại
            ValueError: Nếu file CSV thiếu cột cần thiết
        """
        if not os.path.exists(self.config.dataset_path):
            logger.error(f"File CSV không tồn tại: {self.config.dataset_path}")
            raise FileNotFoundError(f"File CSV không tồn tại: {self.config.dataset_path}")
        
        try:
            df = pd.read_csv(self.config.dataset_path)
            required_columns = ['Message', 'Category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"File CSV thiếu cột: {missing_columns}")
                raise ValueError(f"File CSV thiếu cột: {missing_columns}")
            
            messages = df['Message'].values.tolist()
            labels = df['Category'].values.tolist()
            self.label_encoder.fit(labels)
            
            # Áp dụng preprocess cho messages
            preprocessed_messages = [self.preprocess_text(msg) for msg in messages]
            
            logger.info(f"Đã tải dữ liệu từ {self.config.dataset_path}. Số mẫu: {len(messages)}")
            return preprocessed_messages, labels
        except Exception as e:
            logger.error(f"Lỗi khi đọc file CSV: {str(e)}")
            raise
    
    def create_metadata(self, 
                       messages: List[str], 
                       labels: List[str], 
                       encoded_labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tạo metadata cho mỗi document.
        
        Args:
            messages: Danh sách tin nhắn (đã preprocess)
            labels: Danh sách nhãn gốc
            encoded_labels: Nhãn đã được encode
            
        Returns:
            Danh sách metadata
        """
        metadata = []
        for i, (message, label) in enumerate(zip(messages, labels)):
            metadata.append({
                'index': i,
                'message': message,  # Đây là message đã preprocess
                'label': label,
                'label_encoded': encoded_labels[i]
            })
        
        return metadata
    
    def split_data(self, 
                   messages: List[str], 
                   labels: List[str]) -> Tuple[np.ndarray, ...]:
        """
        Chia dữ liệu thành train và test set.
        
        Args:
            messages: Danh sách tin nhắn
            labels: Danh sách nhãn
            
        Returns:
            Tuple chứa các indices và arrays đã chia
        """
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Chia dữ liệu
        train_indices, test_indices = train_test_split(
            range(len(messages)),
            test_size=self.config.test_size,
            stratify=encoded_labels,  # Chia tỉ lệ spam/ham xấp xỉ tỉ lệ gốc
            random_state=self.config.random_state
        )
        
        return (train_indices, test_indices, 
                encoded_labels[train_indices], encoded_labels[test_indices])
    
    def get_class_names(self) -> np.ndarray:
        """
        Lấy tên các lớp.
        
        Returns:
            Array chứa tên các lớp
        """
        return self.label_encoder.classes_