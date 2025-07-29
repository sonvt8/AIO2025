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
from tfidf_classifier import TFIDFClassifier  # Thêm import cho TF-IDF

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
                          test_embeddings: np.ndarray = None,  # Chỉ cho KNN
                          test_metadata: List[Dict[str, Any]] = None,  # Chỉ cho KNN
                          knn_classifier: KNNClassifier = None,  # KNN classifier
                          tfidf_classifier: TFIDFClassifier = None,  # Thêm TF-IDF classifier
                          k_values: List[int] = None) -> Tuple[Dict, Dict]:
        """
        Đánh giá độ chính xác cho cả KNN và TF-IDF.
        
        Args:
            test_embeddings: Embeddings của test set (cho KNN)
            test_metadata: Metadata của test set (cho KNN)
            knn_classifier: KNN classifier đã train
            tfidf_classifier: TFIDF classifier đã train
            k_values: Danh sách các giá trị k cần test
            
        Returns:
            Tuple chứa kết quả các chỉ số và errors
        """
        if k_values is None:
            k_values = self.config.k_values
            
        knn_results = {}
        knn_errors = {}
        knn_confusion_matrices = {}
        
        true_labels = [meta['label'] for meta in test_metadata]
        
        for k in k_values:
            correct = 0
            predictions = []
            errors = []
            
            for i in tqdm(range(len(test_embeddings)), desc=f"Đánh giá k={k}"):
                query_embedding = test_embeddings[i:i+1].astype('float32')
                true_label = test_metadata[i]['label']
                true_message = test_metadata[i]['message']
                
                pred, neighbors = knn_classifier.predict(query_embedding, k=k)
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
            
            knn_results[k] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            knn_errors[k] = errors
            knn_confusion_matrices[k] = cm
            
            logger.info(f"Độ chính xác với k={k}: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            logger.info(f"Số lỗi với k={k}: {len(errors)}/{len(test_embeddings)} "
                        f"({(len(errors)/len(test_embeddings))*100:.2f}%)")
        
        # Chọn best K dựa trên max weighted F1
        best_k = max(knn_results, key=lambda k: knn_results[k]['f1'])
        logger.info(f"Best K: {best_k} dựa trên F1-score cao nhất ({knn_results[best_k]['f1']:.4f})")
        
        # TF-IDF evaluation
        tfidf_results = {}
        tfidf_predictions = []
        tfidf_errors = []
        messages, labels = self.data_loader.load_data()
        train_indices, test_indices, _, _ = self.data_loader.split_data(messages, labels)
        test_messages = [messages[i] for i in test_indices]
        true_labels = [labels[i] for i in test_indices]  # Reuse true_labels
        
        for i, msg in tqdm(enumerate(test_messages), desc="Đánh giá TF-IDF"):
            pred = tfidf_classifier.predict(msg)['prediction']
            tfidf_predictions.append(pred)
            if pred != true_labels[i]:
                tfidf_errors.append({'index': i, 'message': msg, 'true_label': true_labels[i], 'predicted_label': pred})
        
        tfidf_accuracy = sum(p == t for p, t in zip(tfidf_predictions, true_labels)) / len(true_labels)
        tfidf_precision, tfidf_recall, tfidf_f1, _ = precision_recall_fscore_support(
            true_labels, tfidf_predictions, average='weighted', labels=self.data_loader.get_class_names()
        )
        tfidf_cm = confusion_matrix(true_labels, tfidf_predictions, labels=self.data_loader.get_class_names())
        
        tfidf_results = {'accuracy': tfidf_accuracy, 'precision': tfidf_precision, 'recall': tfidf_recall, 'f1': tfidf_f1}
        logger.info(f"TF-IDF: Accuracy {tfidf_accuracy:.4f}, Precision {tfidf_precision:.4f}, Recall {tfidf_recall:.4f}, F1 {tfidf_f1:.4f}")
        
        # Combine cho viz
        combined_results = {'knn': knn_results, 'best_k': best_k, 'tfidf': tfidf_results}
        combined_cms = {'knn': knn_confusion_matrices, 'tfidf': tfidf_cm}
        
        # Lưu error analysis (thêm TF-IDF)
        self.save_error_analysis(knn_results, knn_errors, len(true_labels), tfidf_results, tfidf_errors)
        
        # Visualize
        self._visualize_results(combined_results, combined_cms, messages, labels, k_values)
        
        return combined_results, knn_errors, combined_cms
    
    def save_error_analysis(self, 
                            knn_results: Dict,
                            knn_errors: Dict,
                            test_size: int,
                            tfidf_results: Dict = None,
                            tfidf_errors: List = None) -> None:
        """
        Lưu phân tích lỗi vào file JSON.
        
        Args:
            knn_results: Kết quả KNN
            knn_errors: Errors KNN
            test_size: Kích thước test set
            tfidf_results: Kết quả TF-IDF (optional)
            tfidf_errors: Errors TF-IDF (optional)
        """
        error_analysis = {
            'timestamp': datetime.now().isoformat(),
            'model': self.config.model_name,
            'test_size': test_size,
            'knn_results': knn_results,
            'errors_by_k': {
                f'k_{k}': {
                    'total_errors': len(errors),
                    'error_rate': len(errors) / test_size,
                    'errors': errors
                } for k, errors in knn_errors.items()
            }
        }
        
        if tfidf_results:
            error_analysis['tfidf_results'] = tfidf_results
            error_analysis['tfidf_errors'] = {
                'total_errors': len(tfidf_errors),
                'error_rate': len(tfidf_errors) / test_size,
                'errors': tfidf_errors
            }
        
        try:
            with open(self.config.output_file, 'w', encoding='utf-8') as f:
                json.dump(error_analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"Phân tích lỗi đã lưu vào: {self.config.output_file}")
            logger.info("Tóm tắt:")
            for k in error_analysis['errors_by_k']:
                logger.info(f"   {k}: {error_analysis['errors_by_k'][k]['total_errors']} lỗi")
            if tfidf_results:
                logger.info(f"   TF-IDF: {error_analysis['tfidf_errors']['total_errors']} lỗi")
        except Exception as e:
            logger.error(f"Lỗi khi lưu phân tích lỗi: {str(e)}")
            raise
    
    def _visualize_results(self, results: Dict, confusion_matrices: Dict, messages: List[str], labels: List[str], k_values: List[int]) -> None:
        """
        Tạo và lưu các biểu đồ trực quan hóa.
        
        Args:
            results: Kết quả combined (knn, tfidf)
            confusion_matrices: Ma trận nhầm lẫn combined
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
        
        # Tạo lưới subplot với height_ratios để giãn hàng và hspace cho padding
        n_k = len(k_values)
        fig = plt.figure(figsize=(6 * n_k, 22))  # Tăng height tổng để hỗ trợ giãn
        gs = GridSpec(5, n_k, figure=fig, height_ratios=[1.2, 1.2, 1.2, 1.2, 1.2], hspace=0.5)  # height_ratios=1.2 cho mỗi hàng, hspace=0.5
        
        # === Hàng 1: Lineplot cho hiệu suất của top K (KNN) ===
        ax1 = fig.add_subplot(gs[0, :])  # Hàng 0, tất cả cột
        knn_metrics_df = pd.DataFrame([
            {'k': k, 'Metric': metric, 'Value': results['knn'][k][metric]}
            for k in k_values
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        ])
        sns.lineplot(data=knn_metrics_df, x='Metric', y='Value', hue='k', marker='o', ax=ax1)
        ax1.set_title("So sánh các chỉ số theo từng k (KNN)")
        ax1.set_xlabel("Metric")
        ax1.set_ylabel("Score")
        values = knn_metrics_df['Value'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        y_min = max(0.9, mean_val - 1.5 * std_val)
        y_max = min(1.0, mean_val + 1.5 * std_val)
        ax1.set_ylim(y_min, y_max)
        ax1.legend(title='k')
        
        # === Hàng 2: Heatmaps cho từng top K (KNN) ===
        for idx, k in enumerate(k_values):
            ax = fig.add_subplot(gs[1, idx])  # Hàng 1, cột idx
            sns.heatmap(confusion_matrices['knn'][k], annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax)
            ax.set_title(f'Confusion Matrix KNN (k={k})')
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
        for p in ax3.patches:
            ax3.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
        # === Hàng 4: Grouped barplot so sánh TF-IDF vs Best KNN ===
        ax4 = fig.add_subplot(gs[3, :])
        best_k = results['best_k']
        comp_df = pd.DataFrame([
            {'Classifier': 'TF-IDF', 'Metric': m, 'Value': results['tfidf'][m]} for m in ['accuracy', 'precision', 'recall', 'f1']
        ] + [
            {'Classifier': f'Best KNN (k={best_k})', 'Metric': m, 'Value': results['knn'][best_k][m]} for m in ['accuracy', 'precision', 'recall', 'f1']
        ])
        sns.barplot(data=comp_df, x='Metric', y='Value', hue='Classifier', palette='Set1', ax=ax4)
        ax4.set_title("So sánh TF-IDF vs Best KNN")
        ax4.set_ylabel("Score")
        ax4.legend(title='Classifier')
        values = comp_df['Value'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax4.set_ylim(max(0.9, mean_val - 1.5 * std_val), min(1.0, mean_val + 1.5 * std_val))
        
        # === Hàng 5: Heatmap cho TF-IDF ===
        ax5 = fig.add_subplot(gs[4, :])
        sns.heatmap(confusion_matrices['tfidf'], annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax5)
        ax5.set_title('Confusion Matrix (TF-IDF)')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        ax5.set_xticklabels(self.data_loader.get_class_names())
        ax5.set_yticklabels(self.data_loader.get_class_names())
        
        # Lưu biểu đồ gộp với DPI cao
        summary_file = os.path.join(self.config.output_dir, 'evaluation_summary.png')
        plt.savefig(summary_file, dpi=300)
        logger.info(f"Đã lưu biểu đồ gộp vào: {summary_file}")
        plt.close()

    def plot_knn_metrics(self, knn_results: Dict, k_values: List[int]) -> plt.Figure:
        """Trả về fig lineplot metrics KNN theo k."""
        fig, ax = plt.subplots(figsize=(8, 5))
        knn_metrics_df = pd.DataFrame([
            {'k': k, 'Metric': metric, 'Value': knn_results[k][metric]}
            for k in k_values
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        ])
        sns.lineplot(data=knn_metrics_df, x='Metric', y='Value', hue='k', marker='o', ax=ax)
        ax.set_title("So sánh các chỉ số theo từng k (KNN)")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        values = knn_metrics_df['Value'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.set_ylim(max(0.9, mean_val - 1.5 * std_val), min(1.0, mean_val + 1.5 * std_val))
        ax.legend(title='k')
        return fig

    def plot_knn_confusion(self, cm: np.ndarray, k: int) -> plt.Figure:
        """Trả về fig heatmap confusion matrix cho KNN tại k."""
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax)
        ax.set_title(f'Confusion Matrix KNN (k={k})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(self.data_loader.get_class_names())
        ax.set_yticklabels(self.data_loader.get_class_names())
        return fig

    def plot_label_distribution(self, labels: List[str]) -> plt.Figure:
        """Trả về fig barplot phân bố labels."""
        fig, ax = plt.subplots(figsize=(6, 4))
        df_bar = pd.DataFrame({'label': self.data_loader.get_class_names(), 
                            'count': [sum(1 for label in labels if label == cls) 
                                        for cls in self.data_loader.get_class_names()]})
        sns.barplot(data=df_bar, x='label', y='count', palette='Set2', ax=ax)
        ax.set_title("Tổng số email theo từng nhãn")
        ax.set_xlabel("Label")
        ax.set_ylabel("Số lượng email")
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        return fig

    def plot_comparison(self, knn_best: Dict, tfidf_results: Dict, best_k: int) -> plt.Figure:
        """Trả về fig grouped bar so sánh TF-IDF vs best KNN."""
        fig, ax = plt.subplots(figsize=(8, 5))
        comp_df = pd.DataFrame([
            {'Classifier': 'TF-IDF', 'Metric': m, 'Value': tfidf_results[m]} 
            for m in ['accuracy', 'precision', 'recall', 'f1']
        ] + [
            {'Classifier': f'Best KNN (k={best_k})', 'Metric': m, 'Value': knn_best[m]} 
            for m in ['accuracy', 'precision', 'recall', 'f1']
        ])
        sns.barplot(data=comp_df, x='Metric', y='Value', hue='Classifier', palette='Set1', ax=ax)
        ax.set_title("So sánh TF-IDF vs Best KNN")
        ax.set_ylabel("Score")
        ax.legend(title='Classifier')
        values = comp_df['Value'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.set_ylim(max(0.9, mean_val - 1.5 * std_val), min(1.0, mean_val + 1.5 * std_val))
        return fig

    def plot_tfidf_confusion(self, cm: np.ndarray) -> plt.Figure:
        """Trả về fig heatmap confusion matrix cho TF-IDF."""
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=False, ax=ax)
        ax.set_title('Confusion Matrix (TF-IDF)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(self.data_loader.get_class_names())
        ax.set_yticklabels(self.data_loader.get_class_names())
        return fig


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