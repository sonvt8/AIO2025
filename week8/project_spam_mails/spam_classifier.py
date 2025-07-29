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
from tfidf_classifier import TFIDFClassifier

logger = logging.getLogger(__name__)

class SpamClassifierPipeline:
    """Pipeline hoàn chỉnh cho spam classification."""
    
    def __init__(self,
                 config: SpamClassifierConfig = None,
                 classifier_type: str = 'knn'):
        """
        Khởi tạo SpamClassifierPipeline.
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config or SpamClassifierConfig()
        self.data_loader = DataLoader(self.config)
        self.embedding_generator = (
            EmbeddingGenerator(self.config)
            if classifier_type == 'knn' else None
        )
        self.classifier = None
        self.classifier_type = classifier_type
        
    def train(self) -> None:
        """Huấn luyện mô hình với dữ liệu."""
        # Tải dữ liệu
        logger.info("Đang tải dữ liệu...")
        messages, labels = self.data_loader.load_data()
        
        if self.classifier_type == 'knn':
            # Kiểm tra số dòng dataset so với embeddings cache
            emb_file = os.path.join(
                'cache', 'embeddings',
                f"embeddings_{self.config.model_name.replace('/', '_')}.npy"
            )
            dataset_count = len(messages)

            if os.path.exists(emb_file):
                try:
                    cache = np.load(emb_file)
                    cache_count = cache.shape[0]
                    if (cache_count != dataset_count
                            and not self.config.regenerate_embeddings):
                        msg = (
                            f"Số dòng trong dataset ({dataset_count}) "
                            f"không khớp với embeddings cache ({cache_count}). "
                            "Chạy lại với --regenerate để cập nhật."
                        )
                        raise ValueError(msg)
                    if (cache_count != dataset_count
                            and self.config.regenerate_embeddings):
                        logger.info(
                            f"Xóa cache cũ: {emb_file} "
                            "(flag regenerate_embeddings=True)"
                        )
                        os.remove(emb_file)
                except Exception as e:
                    logger.error(f"Lỗi khi kiểm tra cache: {e}")
                    raise

        logger.info(f"Các lớp: {self.data_loader.get_class_names()}")

        # Chia dữ liệu
        train_idx, test_idx, y_train, y_test = (
            self.data_loader.split_data(messages, labels)
        )
        train_msgs = [messages[i] for i in train_idx]
        train_lbls = [labels[i] for i in train_idx]
        
        if self.classifier_type == 'knn':
            # Tạo embeddings
            logger.info(f"Tạo embeddings cho {len(messages)} tin nhắn...")
            embeddings = self.embedding_generator.generate_embeddings(messages)
            logger.info(f"Kích thước embeddings: {embeddings.shape}")

            # Tạo metadata
            encoded = self.data_loader.label_encoder.transform(labels)
            metadata = self.data_loader.create_metadata(
                messages, labels, encoded
            )

            # Chia embeddings và metadata
            train_emb = embeddings[train_idx]
            train_meta = [metadata[i] for i in train_idx]

            logger.info(f"Kích thước tập train: {len(train_emb)}")
            logger.info(f"Phân bố nhãn train: {np.bincount(y_train)}")

            # Tạo và huấn luyện classifier
            self.classifier = KNNClassifier(train_emb.shape[1])
            self.classifier.fit(train_emb, train_meta)

        elif self.classifier_type == 'tfidf':
            self.classifier = TFIDFClassifier()
            self.classifier.fit(train_msgs, train_lbls)

        else:
            raise ValueError(
                f"Loại classifier không hợp lệ: {self.classifier_type}"
            )
            
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

        pre_text = self.data_loader.preprocess_text(text)

        if self.classifier_type == 'knn':
            k = k or self.config.default_k
            
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
                logger.info(f"Tin nhắn: {neighbor['message']}")
            
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
            
        if self.classifier_type == 'tfidf':
            return self.classifier.predict(pre_text, return_proba=True)

        raise ValueError(
            f"Loại classifier không hợp lệ: {self.classifier_type}"
        )