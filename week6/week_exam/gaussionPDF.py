import pandas as pd
import numpy as np
from typing import Any, Dict


class GaussianPDF:
    """
    Lớp GaussianPDF tính xác suất theo phân phối chuẩn.
    Dùng pandas để đọc file, NumPy để tính mean và variance.
    """

    def __init__(self) -> None:
        """
        Khởi tạo dict lưu trữ thống kê theo lớp.
        """
        self.class_stats: Dict[Any, Dict[str, float]] = {}
        self.mean_by_class: Dict[Any, float] = {}
        self.variance_by_class: Dict[Any, float] = {}

    def gaussian_pdf(self, x: float, class_label: Any) -> float:
        """
        Tính mật độ xác suất chuẩn P(x | class_label).

        Tham số:
            x: giá trị cần tính.
            class_label: nhãn lớp (số hoặc chuỗi).

        Trả về:
            Giá trị PDF (float).
        """
        if class_label not in self.mean_by_class:
            raise ValueError(f"Chưa có mean cho lớp '{class_label}'")
        if class_label not in self.variance_by_class:
            raise ValueError(f"Chưa có variance cho lớp '{class_label}'")

        mu = self.mean_by_class[class_label]
        var = self.variance_by_class[class_label]
        print(f"mu = {mu}")
        print(f"var = {var}")
        sigma = np.sqrt(var)
        return (
            1 / (sigma * np.sqrt(2 * np.pi)) *
            np.exp(-((x - mu) ** 2) / (2 * var))
        )

    def create_training_data(
        self, filepath: str, feature_col: str, class_col: str
    ) -> None:
        """
        Đọc file Excel, tính mean và variance (ddof=1) bằng NumPy.

        Tham số:
            filepath: đường dẫn file .xlsx
            feature_col: tên cột đặc trưng
            class_col: tên cột nhãn lớp

        Kết quả:
            Điền vào self.mean_by_class và self.variance_by_class.
        """
        df = pd.read_excel(filepath)
        values = df[feature_col].to_numpy()
        labels = df[class_col].to_numpy()
        unique_labels = np.unique(labels)

        # Tính thống kê cho từng lớp
        for lbl in unique_labels:
            mask = labels == lbl
            x = values[mask].astype(float)

            mu = np.mean(x)
            var = np.var(x, ddof=0)

            self.mean_by_class[lbl] = mu
            self.variance_by_class[lbl] = var
            self.class_stats[lbl] = {'mean': mu, 'variance': var}

    def print_class_stats(self) -> None:
        """
        In mean và variance của từng lớp.
        """
        for lbl in sorted(self.class_stats, key=str):
            mu = self.mean_by_class[lbl]
            var = self.variance_by_class[lbl]
            print(f"Class '{lbl}': mean = {mu:.4f}, variance = {var:.4f}")
