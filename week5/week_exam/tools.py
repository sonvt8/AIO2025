import numpy as np
from typing import Union


class ClassicTool:
    """Tool kết hợp các yêu cầu để giải bài tập tuần (gián tiếp)"""

    # def __init__(self, embedding_type):

    def vec_len(self, vector: np.ndarray) -> float:
        """Tính độ dài vector"""
        return np.sqrt(np.sum(vector**2))

    def dot_product(
        self, vector1: np.ndarray, vector2: np.ndarray
    ) -> Union[float, int]:
        """Tích 2 vector"""
        return np.sum(i * j for i, j in zip(vector1, vector2))

    def multi_vector(
        self, matrix: np.ndarray, vector: np.ndarray
    ) -> Union[np.ndarray, float, int]:
        """Tích ma trận và vector với matrix là n x m, vector là m """
        n = len(matrix)  # Số hàng
        m = len(matrix[0])  # Số cột

        # Kiểm tra kích thước hợp lệ
        if m != len(vector):
            raise ValueError(
                "Số cột của ma trận phải bằng số hàng của vector.")

        # Khởi tạo vector kết quả với n phần tử
        result = [0] * n

        # Tính toán từng phần tử của vector kết quả
        for i in range(n):
            sum_product = 0
            for j in range(m):
                sum_product += matrix[i][j] * vector[j]
            result[i] = sum_product

        return np.array(result)

    def matrix_multi_matrix(
        self, matrix1: np.ndarray, matrix2: np.ndarray
    ) -> Union[np.ndarray, float, int]:
        """Tích của 2 ma trận với matrix1 là n x m, matrix2 là m x p"""
        n = len(matrix1)  # Số hàng của matrix1
        m = len(matrix1[0])  # Số cột của matrix1
        p = len(matrix2[0])  # Số cột của matrix2

        # Kiểm tra kích thước hợp lệ
        if m != len(matrix2):
            raise ValueError(
                "Số cột của ma trận 1 phải bằng số hàng của ma trận 2.")

        # Khởi tạo ma trận kết quả với kích thước n x p
        result = [[0 for _ in range(p)] for _ in range(n)]

        # Tính toán từng phần tử của ma trận kết quả
        for i in range(n):
            for j in range(p):
                sum_product = 0
                for k in range(m):
                    sum_product += matrix1[i][k] * matrix2[k][j]
                result[i][j] = sum_product

        return np.array(result)


class QuickTool:
    """Tool kết hợp các yêu cầu để giải bài tập tuần (trực tiếp)"""

    # def __init__(self, embedding_type):

    def vec_len(self, vector: np.ndarray) -> float:
        return np.linalg.norm(vector)

    def dot_product(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> Union[np.ndarray, float, int]:
        return np.dot(vec1, vec2)

    def cosine(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> Union[np.ndarray, float, int]:
        flat1 = np.array(vec1).flatten()
        flat2 = np.array(vec2).flatten()

        result = self.dot_product(flat1, flat2) / (
            (self.vec_len(vec1) * self.vec_len(vec2))
        )
        return result
