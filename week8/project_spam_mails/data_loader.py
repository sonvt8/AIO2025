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
    
    def load_emails_from_folders(self) -> Tuple[List[str], List[str]]:
        """
        Đọc email từ thư mục inbox (ham) và spam, trả về messages và labels.

        Returns:
            Tuple chứa danh sách tin nhắn và nhãn
        """
        messages = []
        labels = []

        # Đọc từ thư mục inbox (ham)
        inbox_path = self.config.inbox_local_dir
        if os.path.exists(inbox_path):
            for filename in os.listdir(inbox_path):
                if filename.endswith('.txt'):
                    try:
                        with open(os.path.join(inbox_path, filename), 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:  # Chỉ thêm nếu không rỗng
                                messages.append(content)
                                labels.append('ham')
                    except Exception as e:
                        logger.warning(f"Lỗi khi đọc file {filename} trong inbox: {str(e)}")

        # Đọc từ thư mục spam
        spam_path = self.config.spam_local_dir
        if os.path.exists(spam_path):
            for filename in os.listdir(spam_path):
                if filename.endswith('.txt'):
                    try:
                        with open(os.path.join(spam_path, filename), 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:  # Chỉ thêm nếu không rỗng
                                messages.append(content)
                                labels.append('spam')
                    except Exception as e:
                        logger.warning(f"Lỗi khi đọc file {filename} trong spam: {str(e)}")

        logger.info(f"Đã đọc {len(messages)} email từ thư mục inbox/spam")
        return messages, labels
    
    def merge_emails_to_dataset(self) -> None:
        """
        Gộp email từ thư mục inbox/spam vào dataset hiện tại, loại bỏ trùng lặp,
        và lưu vào dataset_path. Log so sánh số lượng và tỷ lệ spam/ham trước/sau.
        """
        # Đọc dataset hiện tại
        old_count = 0
        old_spam_ratio = 0.0
        old_ham_ratio = 0.0
        if os.path.exists(self.config.dataset_path):
            try:
                df = pd.read_csv(self.config.dataset_path)
                old_count = len(df)
                if old_count > 0:
                    old_spam_count = len(df[df['Category'] == 'spam'])
                    old_ham_count = len(df[df['Category'] == 'ham'])
                    old_spam_ratio = old_spam_count / old_count
                    old_ham_ratio = old_ham_count / old_count
                existing_messages = set(df['Message'].values)
            except Exception as e:
                logger.error(f"Lỗi khi đọc dataset hiện tại: {str(e)}")
                df = pd.DataFrame(columns=['Category', 'Message'])
                existing_messages = set()
        else:
            df = pd.DataFrame(columns=['Category', 'Message'])
            existing_messages = set()

        # Log thông tin trước khi gộp
        logger.info(f"Trước khi gộp: {old_count} mẫu, tỷ lệ spam: {old_spam_ratio:.2%}, ham: {old_ham_ratio:.2%}")

        # Đọc email từ thư mục
        new_messages, new_labels = self.load_emails_from_folders()

        # Gộp dữ liệu mới, loại bỏ trùng lặp
        new_data = []
        for msg, label in zip(new_messages, new_labels):
            if msg not in existing_messages:
                new_data.append({'Category': label, 'Message': msg})
                existing_messages.add(msg)

        if new_data:
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Lưu dataset mới
            try:
                df.to_csv(self.config.dataset_path, index=False, encoding='utf-8')
                new_count = len(df)
                new_spam_count = len(df[df['Category'] == 'spam'])
                new_ham_count = len(df[df['Category'] == 'ham'])
                new_spam_ratio = new_spam_count / new_count if new_count > 0 else 0.0
                new_ham_ratio = new_ham_count / new_count if new_count > 0 else 0.0
                
                # Log thông tin sau khi gộp
                logger.info(f"Đã gộp {len(new_data)} email mới vào dataset: {self.config.dataset_path}")
                logger.info(f"Sau khi gộp: {new_count} mẫu, tỷ lệ spam: {new_spam_ratio:.2%}, ham: {new_ham_ratio:.2%}")
            except Exception as e:
                logger.error(f"Lỗi khi lưu dataset: {str(e)}")
                raise
        else:
            logger.info("Không có email mới để gộp hoặc tất cả đã trùng lặp")

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
        if not hasattr(self.label_encoder, 'classes_'):
            # Tải nhãn gốc từ file CSV
            _, labels = self.load_data()
            self.label_encoder.fit(labels)
        return self.label_encoder.classes_