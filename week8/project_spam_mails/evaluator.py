"""
Module đánh giá hiệu suất của mô hình.
"""
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from knn_classifier import KNNClassifier
from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from spam_classifier import SpamClassifierPipeline

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class đánh giá hiệu suất mô hình."""
    
    def __init__(self, config: SpamClassifierConfig):
        """
        Khởi tạo ModelEvaluator.
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.embedding_generator = EmbeddingGenerator(config)
    
    def evaluate_accuracy(self, 
                         test_embeddings: np.ndarray,
                         test_metadata: List[Dict[str, Any]],
                         classifier: KNNClassifier,
                         k_values: List[int] = None) -> Tuple[Dict, Dict]:
        """
        Đánh giá độ chính xác và các chỉ số khác với các giá trị k.
        
        Args:
            test_embeddings: Embeddings của test set
            test_metadata: Metadata của test set
            classifier: KNN classifier đã train
            k_values: Danh sách các giá trị k cần test
            
        Returns:
            Tuple chứa kết quả các chỉ số và errors
        """
        if k_values is None:
            k_values = self.config.k_values
            
        results = {}
        all_errors = {}
        confusion_matrices = {}
        
        true_labels = [meta['label'] for meta in test_metadata]
        
        for k in k_values:
            correct = 0
            predictions = []
            errors = []
            
            for i in tqdm(range(len(test_embeddings)), desc=f"Đánh giá k={k}"):
                query_embedding = test_embeddings[i:i+1].astype('float32')
                true_label = test_metadata[i]['label']
                true_message = test_metadata[i]['message']
                
                pred, neighbors = classifier.predict(query_embedding, k=k)
                predictions.append(pred)
                
                if pred == true_label:
                    correct += 1
                else:
                    error_info = {
                        'index': i,
                        'original_index': test_metadata[i]['index'],
                        'message': true_message,
                        'true_label': true_label,
                        'predicted_label': pred,
                        'neighbors': neighbors,
                        'label_distribution': {
                            label: sum(1 for n in neighbors if n['label'] == label)
                            for label in set(n['label'] for n in neighbors)
                        }
                    }
                    errors.append(error_info)
            
            # Tính toán các chỉ số
            accuracy = correct / len(test_embeddings)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', labels=self.data_loader.get_class_names()
            )
            cm = confusion_matrix(true_labels, predictions, labels=self.data_loader.get_class_names())
            
            results[k] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            all_errors[k] = errors
            confusion_matrices[k] = cm
            
            logger.info(f"Độ chính xác với k={k}: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            logger.info(f"Số lỗi với k={k}: {len(errors)}/{len(test_embeddings)} "
                        f"({(len(errors)/len(test_embeddings))*100:.2f}%)")
        
        # Lưu phân tích lỗi
        self.save_error_analysis(results, all_errors, len(test_embeddings))
        
        # Trực quan hóa
        messages, labels = self.data_loader.load_data()
        self._visualize_results(results, confusion_matrices, messages, labels, k_values)
        
        return results, all_errors
    
    def save_error_analysis(self, 
                           accuracy_results: Dict,
                           error_results: Dict,
                           test_size: int) -> None:
        """
        Lưu phân tích lỗi vào file JSON.
        
        Args:
            accuracy_results: Kết quả các chỉ số
            error_results: Kết quả errors
            test_size: Kích thước test set
        """
        error_analysis = {
            'timestamp': datetime.now().isoformat(),
            'model': self.config.model_name,
            'test_size': test_size,
            'results': accuracy_results,
            'errors_by_k': {
                f'k_{k}': {
                    'total_errors': len(errors),
                    'error_rate': len(errors) / test_size,
                    'errors': errors
                } for k, errors in error_results.items()
            }
        }
        
        try:
            with open(self.config.output_file, 'w', encoding='utf-8') as f:
                json.dump(error_analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"Phân tích lỗi đã lưu vào: {self.config.output_file}")
            logger.info("Tóm tắt:")
            for k, errors in error_results.items():
                logger.info(f"   k={k}: {len(errors)} lỗi trong {test_size} mẫu")
        except Exception as e:
            logger.error(f"Lỗi khi lưu phân tích lỗi: {str(e)}")
            raise
    
    def _visualize_results(self, results: Dict, confusion_matrices: Dict, messages: List[str], labels: List[str], k_values: List[int]) -> None:
        """
        Tạo và lưu các biểu đồ trực quan hóa.
        
        Args:
            results: Kết quả các chỉ số
            confusion_matrices: Ma trận nhầm lẫn
            messages: Danh sách tin nhắn
            labels: Danh sách nhãn
            k_values: Danh sách các giá trị k
        """
        # Kiểm tra dữ liệu đầu vào
        if not results or not confusion_matrices or not labels:
            logger.error("Dữ liệu đầu vào cho trực quan hóa rỗng.")
            return

        # Sử dụng style hợp lệ của Matplotlib
        plt.style.use('seaborn-v0_8')
        
        # Tạo lưới subplot cho biểu đồ gộp (3 hàng, số cột = số k_values)
        n_k = len(k_values)
        fig = plt.figure(figsize=(6 * n_k, 14))
        gs = GridSpec(3, n_k, figure=fig)
        
        # === Hàng 1: Lineplot cho hiệu suất của top K ===
        ax1 = fig.add_subplot(gs[0, :])  # Hàng 0, tất cả cột
        metrics_df = pd.DataFrame([
            {'k': k, 'Metric': metric, 'Value': results[k][metric]}
            for k in k_values
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        ])
        sns.lineplot(data=metrics_df, x='Metric', y='Value', hue='k', marker='o', ax=ax1)
        ax1.set_title("So sánh các chỉ số theo từng k")
        ax1.set_xlabel("Metric")
        ax1.set_ylabel("Score")
        # Tính phạm vi trục Y dựa trên giá trị trung bình và độ lệch chuẩn
        values = metrics_df['Value'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        y_min = max(0.9, mean_val - 1.5 * std_val)  # Đảm bảo không dưới 0.9
        y_max = min(1.0, mean_val + 1.5 * std_val)  # Đảm bảo không vượt 1.0
        ax1.set_ylim(y_min, y_max)
        ax1.legend(title='k')
        
        # === Hàng 2: Heatmaps cho từng top K ===
        for idx, k in enumerate(k_values):
            ax = fig.add_subplot(gs[1, idx])  # Hàng 1, cột idx
            sns.heatmap(confusion_matrices[k], annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax)
            ax.set_title(f'Confusion Matrix (k={k})')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(self.data_loader.get_class_names())
            ax.set_yticklabels(self.data_loader.get_class_names())
        
        # === Hàng 3: Barplot cho phân bố nhãn ===
        ax3 = fig.add_subplot(gs[2, :])  # Hàng 2, tất cả cột
        df_bar = pd.DataFrame({'label': self.data_loader.get_class_names(), 'count': [sum(1 for label in labels if label == cls) for cls in self.data_loader.get_class_names()]})
        sns.barplot(data=df_bar, x='label', y='count', palette='Set2', ax=ax3)
        ax3.set_title("Tổng số email theo từng nhãn")
        ax3.set_xlabel("Label")
        ax3.set_ylabel("Số lượng email")
        # Thêm số liệu trên đỉnh cột
        for p in ax3.patches:
            ax3.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
        # Lưu biểu đồ gộp với DPI cao
        summary_file = os.path.join(self.config.output_dir, 'evaluation_summary.png')
        plt.savefig(summary_file, dpi=300)
        logger.info(f"Đã lưu biểu đồ gộp vào: {summary_file}")
        plt.close()

def main():
    """Hàm chính để chạy đánh giá mô hình qua command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Run model evaluation with visualization.")
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Run model evaluation with visualization')
    parser.add_argument('--k-values', type=str, default=None,
                        help='Comma-separated list of k values to evaluate (e.g., "1,3,5")')
    parser.add_argument('--regenerate', action='store_true', default=False,
                        help='Regenerate embeddings before evaluation')
    args = parser.parse_args()
    
    if not args.evaluate:
        logger.info("Không chạy đánh giá. Sử dụng --evaluate để kích hoạt.")
        return
    
    # Khởi tạo cấu hình và pipeline
    config = SpamClassifierConfig()
    config.regenerate_embeddings = args.regenerate
    if args.k_values:
        config.k_values = [int(k) for k in args.k_values.split(',')]
    
    # Tải và huấn luyện mô hình
    pipeline = SpamClassifierPipeline(config)
    pipeline.train()
    
    # Chạy đánh giá
    evaluator = ModelEvaluator(config)
    messages, labels = evaluator.data_loader.load_data()
    embeddings = evaluator.embedding_generator.generate_embeddings(messages)
    train_indices, test_indices, y_train, y_test = evaluator.data_loader.split_data(messages, labels)
    test_embeddings = embeddings[test_indices]
    test_metadata = [evaluator.data_loader.create_metadata(messages, labels, evaluator.data_loader.label_encoder.transform(labels))[i] for i in test_indices]
    
    evaluator.evaluate_accuracy(test_embeddings, test_metadata, pipeline.classifier, config.k_values)

if __name__ == "__main__":
    main()