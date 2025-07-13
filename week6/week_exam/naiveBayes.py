import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    """Lớp phân loại Naive Bayes để dự đoán dựa trên xác suất.

    Thuộc tính:
        _train_data (np.ndarray): Dữ liệu huấn luyện.
        _prior_probs (np.ndarray): Xác suất trước (prior probabilities).
        _conditional_probs (list): Xác suất có điều kiện (conditional
                                  probabilities).
        _feature_names (list): Tên hoặc giá trị đặc trưng của các cột.
    """

    def __init__(self) -> None:
        """Khởi tạo lớp NaiveBayesClassifier.

        Không có tham số đầu vào, khởi tạo các thuộc tính rỗng.
        """
        self._class_names = None
        self._train_data = None
        self._prior_probs = None
        self._conditional_probs = None
        self._feature_names = None


    def create_training_data(self, file_path: str) -> np.ndarray:
        """
        Đọc dữ liệu huấn luyện từ file Excel hoặc CSV.
        Args:
            file_path (str): Đường dẫn tới file dữ liệu (.csv hoặc .xlsx).
        Returns:
            np.ndarray: Mảng dữ liệu huấn luyện (không bao gồm header).
        """
        # Đọc file dữ liệu train
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Chỉ hỗ trợ file .csv hoặc .xlsx")
        
        # Loại bỏ header và chuyển sang numpy array
        data = df.values
        return np.array(data)

    def compute_prior_probabilities(self, train_data: np.ndarray) -> np.ndarray:
        """Tính xác suất trước dựa trên dữ liệu huấn luyện.

        Args:
            train_data (np.ndarray): Dữ liệu huấn luyện.

        Returns:
            np.ndarray: Mảng chứa xác suất trước cho mỗi lớp.
        """
        total_samples = len(train_data)
        prior_probs = np.zeros(len(self._class_names))
        for i, class_name in enumerate(self._class_names):
            class_count = np.sum(train_data[:, -1] == class_name)
            prior_probs[i] = class_count / total_samples
        return prior_probs

    def compute_conditional_probabilities(
        self, train_data: np.ndarray
    ) -> tuple[list, list]:
        """Tính xác suất có điều kiện cho từng đặc trưng.

        Args:
            train_data (np.ndarray): Dữ liệu huấn luyện.

        Returns:
            tuple[list, list]: Cặp (xác suất có điều kiện, giá trị đặc trưng).
        """
        n_features = train_data.shape[1] - 1
        conditional_probs = []
        feature_values = []

        for feature_idx in range(n_features):
            unique_values = np.unique(train_data[:, feature_idx])
            feature_values.append(unique_values)
            feature_cond_probs = np.zeros((len(self._class_names), len(unique_values)))

            for class_idx, class_name in enumerate(self._class_names):
                class_mask = train_data[:, -1] == class_name
                class_samples = train_data[class_mask]

                for value_idx, value in enumerate(unique_values):
                    feature_count = np.sum(class_samples[:, feature_idx] == value)
                    feature_cond_probs[class_idx, value_idx] = (
                        feature_count / len(class_samples)
                    )

            conditional_probs.append(feature_cond_probs)
        return conditional_probs, feature_values

    def train(self, train_data: np.ndarray) -> None:
        """Huấn luyện mô hình Naive Bayes.

        Args:
            train_data (np.ndarray): Dữ liệu huấn luyện.

        Lưu trữ xác suất trước và có điều kiện vào thuộc tính lớp.
        """
        self._train_data = train_data
        self._class_names = np.unique(train_data[:, -1]).tolist()
        self._prior_probs, self._conditional_probs, self._feature_names = (
            self.compute_prior_probabilities(train_data),
            self.compute_conditional_probabilities(train_data)[0],
            self.compute_conditional_probabilities(train_data)[1],
        )

    def get_feature_index(
        self, feature_value: str, feature_values: np.ndarray
    ) -> int:
        """Lấy chỉ số của giá trị đặc trưng trong mảng giá trị.

        Args:
            feature_value (str): Giá trị cần tìm.
            feature_values (np.ndarray): Mảng chứa các giá trị đặc trưng.

        Returns:
            int: Chỉ số của giá trị đặc trưng.
        """
        return np.where(feature_values == feature_value)[0][0]

    def predict_tennis(
        self, X: list[str]
    ) -> tuple[str, dict[str, float]]:
        """Dự đoán kết quả chơi tennis dựa trên đặc trưng.

        Args:
            X (list[str]): Danh sách các giá trị đặc trưng.

        Returns:
            tuple[str, dict[str, float]]: Cặp (dự đoán, từ điển xác suất).
        """

        # Lấy chỉ số đặc trưng
        feature_indices = []
        for i, feature_value in enumerate(X):
            feature_indices.append(
                self.get_feature_index(feature_value, self._feature_names[i])
            )

        # Tính xác suất cho mỗi lớp
        class_probabilities = []
        for class_idx in range(len(self._class_names)):
            prob = self._prior_probs[class_idx]
            for feature_idx, value_idx in enumerate(feature_indices):
                prob *= self._conditional_probs[feature_idx][
                    class_idx, value_idx
                ]
            class_probabilities.append(prob)

        # Chuẩn hóa xác suất
        total_prob = sum(class_probabilities)
        if total_prob > 0:
            normalized_probs = [
                p / total_prob for p in class_probabilities
            ]
        else:
            normalized_probs = [0.5, 0.5]

        # Dự đoán
        predicted_class_idx = np.argmax(class_probabilities)
        prediction = self._class_names[predicted_class_idx]

        # Tạo từ điển xác suất
        prob_dict = {
            'No': round(normalized_probs[0].item(), 2),
            'Yes': round(normalized_probs[1].item(), 2),
        }

        return prediction, prob_dict, class_probabilities
