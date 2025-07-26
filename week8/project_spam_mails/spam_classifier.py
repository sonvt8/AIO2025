"""
Pipeline chính cho spam classification.
"""
import numpy as np
from typing import Dict, Any
from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from knn_classifier import KNNClassifier
from evaluator import ModelEvaluator


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
        self.evaluator = ModelEvaluator(self.config)
        
    def train(self) -> None:
        """Huấn luyện mô hình với dữ liệu."""
        # Tải dữ liệu
        print("Đang tải dữ liệu...")
        messages, labels = self.data_loader.load_data()
        
        print(f'Các lớp: {self.data_loader.get_class_names()}')
        
        # Tạo embeddings
        print(f"Đang tạo embeddings cho {len(messages)} tin nhắn...")
        embeddings = self.embedding_generator.generate_embeddings(messages)
        print(f"Kích thước embeddings: {embeddings.shape}")
        
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
        test_embeddings = embeddings[test_indices]
        train_metadata = [metadata[i] for i in train_indices]
        test_metadata = [metadata[i] for i in test_indices]
        
        # print(f"Kích thước train: {len(train_embeddings)}")
        # print(f"Kích thước test: {len(test_embeddings)}")
        # print(f"Phân bố nhãn train: {np.bincount(y_train)}")
        # print(f"Phân bố nhãn test: {np.bincount(y_test)}")
        
        # Tạo và huấn luyện classifier
        self.classifier = KNNClassifier(train_embeddings.shape[1])
        self.classifier.fit(train_embeddings, train_metadata)
        
        # Đánh giá mô hình
        print("Đang đánh giá độ chính xác trên test set...")
        accuracy_results, error_results = self.evaluator.evaluate_accuracy(
            test_embeddings, test_metadata, self.classifier
        )
        
        # # Hiển thị kết quả
        # print("\n" + "="*50)
        # print("KẾT QUẢ ĐỘ CHÍNH XÁC")
        # print("="*50)
        # for k, accuracy in accuracy_results.items():
        #     print(f"Top-{k} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        # print("="*50)
        
        # Lưu phân tích lỗi
        self.evaluator.save_error_analysis(
            accuracy_results, error_results, len(test_embeddings)
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
        
        if k is None:
            k = self.config.default_k
        
        # print(f"\n***Đang phân loại: '{text}'")
        # print(f"\n***Sử dụng top-{k} nearest neighbors")
        
        # Tạo query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(
            text
        )
        
        # Phân loại
        prediction, neighbors = self.classifier.predict(
            query_embedding, k=k
        )
        
        # Hiển thị kết quả
        # print(f"\n***Dự đoán: {prediction.upper()}")
        # print("\n***Top neighbors:")
        # for i, neighbor in enumerate(neighbors, 1):
        #     print(f"{i}. Nhãn: {neighbor['label']} | "
        #           f"Điểm: {neighbor['score']:.4f}")
        #     print(f"   Tin nhắn: {neighbor['message']}")
        
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
