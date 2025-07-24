"""
Module xử lý và tải dữ liệu cho spam classification.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
        
    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Tải dữ liệu từ file CSV.
        
        Returns:
            Tuple chứa danh sách tin nhắn và nhãn
        """
        df = pd.read_csv(self.config.dataset_path)
        messages = df['Message'].values.tolist()
        labels = df['Category'].values.tolist()
        self.label_encoder.fit(labels)
        
        return messages, labels
    
    def create_metadata(self, 
                       messages: List[str], 
                       labels: List[str], 
                       encoded_labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tạo metadata cho mỗi document.
        
        Args:
            messages: Danh sách tin nhắn
            labels: Danh sách nhãn gốc
            encoded_labels: Nhãn đã được encode
            
        Returns:
            Danh sách metadata
        """
        metadata = []
        for i, (message, label) in enumerate(zip(messages, labels)):
            metadata.append({
                'index': i,
                'message': message,
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
            stratify=encoded_labels, #chia tỉ lệ spam/ham xấp xỉ tỉ lệ gốc
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
