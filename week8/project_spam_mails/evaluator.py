"""
Module đánh giá hiệu suất của mô hình.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from collections import Counter
import json
from knn_classifier import KNNClassifier
from datetime import datetime


class ModelEvaluator:
    """Class đánh giá hiệu suất mô hình."""
    
    def __init__(self, config):
        """
        Khởi tạo ModelEvaluator.
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config
    
    def evaluate_accuracy(self, 
                         test_embeddings: np.ndarray,
                         test_metadata: List[Dict[str, Any]],
                         classifier: KNNClassifier,
                         k_values: List[int] = None) -> Tuple[Dict, Dict]:
        """
        Đánh giá độ chính xác với các giá trị k khác nhau.
        
        Args:
            test_embeddings: Embeddings của test set
            test_metadata: Metadata của test set
            classifier: KNN classifier đã train
            k_values: Danh sách các giá trị k cần test
            
        Returns:
            Tuple chứa kết quả accuracy và errors
        """
        if k_values is None:
            k_values = self.config.k_values
            
        results = {}
        all_errors = {}
        
        for k in k_values:
            correct = 0
            total = len(test_embeddings)
            errors = []
            
            for i in tqdm(range(total), desc=f"Đánh giá k={k}"):
                query_embedding = test_embeddings[i:i+1].astype('float32')
                true_label = test_metadata[i]['label']
                true_message = test_metadata[i]['message']
                
                # Tìm kiếm trong FAISS index
                scores, indices = classifier.index.search(
                    query_embedding, k
                )
                
                # Lấy predictions từ top-k neighbors
                predictions = []
                neighbor_details = []
                for j in range(k):
                    neighbor_idx = indices[0][j]
                    neighbor_data = classifier.train_metadata[neighbor_idx]
                    neighbor_score = float(scores[0][j])
                    
                    predictions.append(neighbor_data['label'])
                    neighbor_details.append({
                        'label': neighbor_data['label'],
                        'message': neighbor_data['message'],
                        'score': neighbor_score
                    })
                
                # Majority vote
                unique_labels, counts = np.unique(
                    predictions, return_counts=True
                )
                predicted_label = unique_labels[np.argmax(counts)]
                
                if predicted_label == true_label:
                    correct += 1
                else:
                    # Thu thập thông tin lỗi
                    error_info = {
                        'index': i,
                        'original_index': test_metadata[i]['index'],
                        'message': true_message,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'neighbors': neighbor_details,
                        'label_distribution': {
                            label: int(count) 
                            for label, count in zip(unique_labels, counts)
                        }
                    }
                    errors.append(error_info)
            
            accuracy = correct / total
            error_count = total - correct
            
            results[k] = accuracy
            all_errors[k] = errors
            
            print(f"Độ chính xác với k={k}: {accuracy:.4f}")
            print(f"Số lỗi với k={k}: {error_count}/{total} "
                  f"({(error_count/total)*100:.2f}%)")
        
        return results, all_errors
    
    def save_error_analysis(self, 
                           accuracy_results: Dict,
                           error_results: Dict,
                           test_size: int) -> None:
        """
        Lưu phân tích lỗi vào file JSON.
        
        Args:
            accuracy_results: Kết quả accuracy
            error_results: Kết quả errors
            test_size: Kích thước test set
        """
        error_analysis = {
            'timestamp': datetime.now().isoformat(),
            'model': self.config.model_name,
            'test_size': test_size,
            'accuracy_results': accuracy_results,
            'errors_by_k': {}
        }
        
        for k, errors in error_results.items():
            error_analysis['errors_by_k'][f'k_{k}'] = {
                'total_errors': len(errors),
                'error_rate': len(errors) / test_size,
                'errors': errors
            }
        
        # Lưu vào file JSON
        with open(self.config.output_file, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n***Phân tích lỗi đã lưu vào: {self.config.output_file}***")
        print("\n***Tóm tắt:")
        for k, errors in error_results.items():
            print(f"   k={k}: {len(errors)} lỗi trong {test_size} mẫu")
