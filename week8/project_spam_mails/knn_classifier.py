"""
Module phân loại sử dụng K-Nearest Neighbors với FAISS.
"""
import faiss
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import Counter


class KNNClassifier:
    """Class phân loại KNN sử dụng FAISS."""
    
    def __init__(self, embedding_dim: int):
        """
        Khởi tạo KNN Classifier.
        
        Args:
            embedding_dim: Số chiều của embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product
        self.train_metadata = None
    
    def fit(self, 
            train_embeddings: np.ndarray, 
            train_metadata: List[Dict[str, Any]]) -> None:
        """
        Huấn luyện classifier với dữ liệu train.
        
        Args:
            train_embeddings: Embeddings của dữ liệu train
            train_metadata: Metadata của dữ liệu train
        """
        self.train_metadata = train_metadata
        self.index.add(train_embeddings.astype('float32'))
        print(f"FAISS index đã tạo với {self.index.ntotal} vectors")
    
    def predict(self, 
                query_embedding: np.ndarray, 
                k: int = 1) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Phân loại một query sử dụng k-nearest neighbors.
        
        Args:
            query_embedding: Embedding của query
            k: Số lượng neighbors
            
        Returns:
            Tuple chứa prediction và thông tin neighbors
        """
        # Tìm kiếm trong FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Lấy predictions từ top-k neighbors
        predictions = []
        neighbor_info = []
        
        for i in range(k):
            neighbor_idx = indices[0][i]
            neighbor_score = scores[0][i]
            neighbor_data = self.train_metadata[neighbor_idx]
            
            predictions.append(neighbor_data['label'])
            neighbor_info.append({
                'score': float(neighbor_score),
                'label': neighbor_data['label'],
                'message': self._truncate_message(neighbor_data['message'])
            })
        
        # Majority vote cho prediction cuối cùng
        final_prediction = Counter(predictions).most_common(1)[0][0]
        
        return final_prediction, neighbor_info
    
    def _truncate_message(self, message: str, max_length: int = 100) -> str:
        """
        Cắt ngắn message để hiển thị.
        
        Args:
            message: Tin nhắn gốc
            max_length: Độ dài tối đa
            
        Returns:
            Tin nhắn đã cắt ngắn
        """
        if len(message) > max_length:
            return message[:max_length] + "..."
        return message
