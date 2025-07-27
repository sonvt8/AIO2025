"""
Mô-đun phân loại spam dựa trên TF-IDF sử dụng scikit-learn.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import List, Dict, Any

class TFIDFClassifier:
    """Bộ phân loại TF-IDF sử dụng Logistic Regression."""
    
    def __init__(self, max_features: int = 5000):
        """
        Khởi tạo bộ phân loại.
        
        Args:
            max_features: Số đặc trưng tối đa cho TF-IDF (giới hạn kích thước từ vựng để tăng tốc).
        """
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features)),  # Biến đổi văn bản thành TF-IDF
            ('clf', LogisticRegression(max_iter=1000))  # Logistic Regression cho phân loại nhị phân
        ])
    
    def fit(self, train_texts: List[str], train_labels: List[str]) -> None:
        """
        Huấn luyện mô hình trên dữ liệu.
        
        Args:
            train_texts: Danh sách tin nhắn đã được tiền xử lý.
            train_labels: Danh sách nhãn ('spam' hoặc 'ham').
        """
        self.model.fit(train_texts, train_labels)
    
    def predict(self, text: str, return_proba: bool = False) -> Dict[str, Any]:
        """
        Dự đoán nhãn cho một văn bản.
        
        Args:
            text: Văn bản đầu vào (tiền xử lý trước khi gọi).
            return_proba: Nếu True, trả về cả xác suất.
        
        Returns:
            Dict chứa 'prediction' và tùy chọn 'probabilities'.
        """
        pred = self.model.predict([text])[0]
        result = {'prediction': pred}
        if return_proba:
            proba = self.model.predict_proba([text])[0]
            result['probabilities'] = dict(zip(self.model.classes_, proba))
        return result
