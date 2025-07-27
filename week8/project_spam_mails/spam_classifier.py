"""
Pipeline chính cho spam classification.
"""
import numpy as np
import os
import logging
from typing import Dict, Any
from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from knn_classifier import KNNClassifier

logger = logging.getLogger(__name__)

class SpamClassifierPipeline:
    """Pipeline hoàn chỉnh cho spam classification."""
    
    def __init__(self, config: SpamClassifierConfig = None):
        """
        Khởi tạo SpamClassifierPipeline.
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config or SpamClassifierConfig()
        self.data_loader = DataLoader(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.classifier = None
        
    def train(self) -> None:
        """Huấn luyện mô hình với dữ liệu."""
        # Tải dữ liệu
        logger.info("Đang tải dữ liệu...")
        messages, labels = self.data_loader.load_data()
        
        # Kiểm tra số dòng dataset so với embeddings cache
        embeddings_file = os.path.join('cache', 'embeddings', f"embeddings_{self.config.model_name.replace('/', '_')}.npy")
        dataset_count = len(messages)
        if os.path.exists(embeddings_file):
            try:
                embeddings = np.load(embeddings_file)
                cache_count = embeddings.shape[0]
                if cache_count != dataset_count and not self.config.regenerate_embeddings:
                    raise ValueError(
                        f"Số dòng trong dataset ({dataset_count}) không khớp với embeddings cache ({cache_count}). "
                        "Vui lòng chạy lại với --regenerate để cập nhật embeddings."
                    )
                elif cache_count != dataset_count and self.config.regenerate_embeddings:
                    logger.info(f"Đã xóa embeddings cache cũ: {embeddings_file} (do flag regenerate_embeddings=True)")
                    os.remove(embeddings_file)
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra embeddings cache: {str(e)}")
                raise
        
        logger.info(f'Các lớp: {self.data_loader.get_class_names()}')
        
        # Tạo embeddings
        logger.info(f"Đang tạo embeddings cho {len(messages)} tin nhắn...")
        embeddings = self.embedding_generator.generate_embeddings(messages)
        logger.info(f"Kích thước embeddings: {embeddings.shape}")
        
        # Chia dữ liệu
        (train_indices, test_indices, 
         y_train, y_test) = self.data_loader.split_data(messages, labels)
        
        # Tạo metadata
        encoded_labels = self.data_loader.label_encoder.transform(labels)
        metadata = self.data_loader.create_metadata(
            messages, labels, encoded_labels
        )
        
        # Chia embeddings và metadata
        train_embeddings = embeddings[train_indices]
        train_metadata = [metadata[i] for i in train_indices]
        
        logger.info(f"Kích thước train: {len(train_embeddings)}")
        logger.info(f"Phân bố nhãn train: {np.bincount(y_train)}")
        
        # Tạo và huấn luyện classifier
        self.classifier = KNNClassifier(train_embeddings.shape[1])
        self.classifier.fit(train_embeddings, train_metadata)
    
    def predict(self, text: str, k: int = None) -> Dict[str, Any]:
        """
        Phân loại một văn bản.
        
        Args:
            text: Văn bản cần phân loại
            k: Số lượng neighbors (mặc định từ config)
            
        Returns:
            Dict chứa kết quả phân loại
        """
        if self.classifier is None:
            raise ValueError("Mô hình chưa được huấn luyện. "
                           "Hãy gọi train() trước.")
        
        if k is None:
            k = self.config.default_k
        
        logger.info(f"\n***Đang phân loại: '{text}'")
        logger.info(f"\n***Sử dụng top-{k} nearest neighbors")
        
        # Tạo query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(
            text
        )
        
        # Phân loại
        prediction, neighbors = self.classifier.predict(
            query_embedding, k=k
        )
        
        # Hiển thị kết quả
        logger.info(f"\n***Dự đoán: {prediction.upper()}")
        logger.info("\n***Top neighbors:")
        for i, neighbor in enumerate(neighbors, 1):
            logger.info(f"{i}. Nhãn: {neighbor['label']} | "
                        f"Điểm: {neighbor['score']:.4f}")
            logger.info(f"   Tin nhắn: {neighbor['message']}")
        
        # Đếm phân bố nhãn
        labels = [n['label'] for n in neighbors]
        label_counts = {
            label: labels.count(label) for label in set(labels)
        }
        
        return {
            'prediction': prediction,
            'neighbors': neighbors,
            'label_distribution': label_counts
        }