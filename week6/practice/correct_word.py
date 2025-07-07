import csv
from collections import deque
from typing import Dict, List, Set, Tuple
import nltk
import pandas as pd
from nltk.corpus import words
from collections import Counter

nltk.download("words")


class LevenshteinCorrector:
    """Bộ sửa lỗi chính tả sử dụng khoảng cách Levenshtein."""

    def __init__(self, vocabulary: Set[str], max_suggestions: int = 5, max_distance: int = 2):
        self.vocabulary = vocabulary
        self.max_suggestions = max_suggestions
        self.max_distance = max_distance

    @staticmethod
    def levenshtein_distance(token1: str, token2: str) -> int:
        """Tính khoảng cách Levenshtein giữa hai chuỗi."""
        distances = [[0] * (len(token2) + 1) for _ in range(len(token1) + 1)]
        for t1 in range(len(token1) + 1):
            distances[t1][0] = t1
        for t2 in range(len(token2) + 1):
            distances[0][t2] = t2
        for t1 in range(1, len(token1) + 1):
            for t2 in range(1, len(token2) + 1):
                if token1[t1 - 1] == token2[t2 - 1]:
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    distances[t1][t2] = min(
                        distances[t1][t2 - 1],  # chèn
                        distances[t1 - 1][t2],  # xóa
                        distances[t1 - 1][t2 - 1],  # thay thế
                    ) + 1
        return distances[len(token1)][len(token2)]

    def suggest_corrections(self, token: str) -> List[str]:
        """Gợi ý từ sửa lỗi dựa trên khoảng cách Levenshtein."""
        token_lower = token.lower()
        if token_lower in self.vocabulary:
            return [token]

        candidates: List[Tuple[str, int]] = []
        for word in self.vocabulary:
            if abs(len(token_lower) - len(word)) <= self.max_distance:
                dist = self.levenshtein_distance(token_lower, word)
                if dist <= self.max_distance:
                    candidates.append((word, dist))

        candidates.sort(key=lambda x: x[1])
        return [word for word, _ in candidates[: self.max_suggestions]]


class KeyboardLevenshteinCorrector:
    """Bộ sửa lỗi chính tả kết hợp khoảng cách Levenshtein và bàn phím."""

    def __init__(
        self,
        vocabulary: Set[str],
        keyboard_distances: Dict[str, Dict[str, int]],
        max_suggestions: int = 5,
        max_distance: float = 2.0,
    ):
        self.vocabulary = vocabulary
        self.keyboard_distances = keyboard_distances
        self.max_suggestions = max_suggestions
        self.max_distance = max_distance
        self.cost_insert = 1.0
        self.cost_delete = 1.0

    def keyboard_levenshtein_distance(self, token1: str, token2: str) -> float:
        """Tính khoảng cách Levenshtein có trọng số khoảng cách bàn phím."""
        token1, token2 = token1.lower(), token2.lower()
        distances = [[0.0] * (len(token2) + 1) for _ in range(len(token1) + 1)]

        for t1 in range(len(token1) + 1):
            distances[t1][0] = t1 * self.cost_delete
        for t2 in range(len(token2) + 1):
            distances[0][t2] = t2 * self.cost_insert

        for t1 in range(1, len(token1) + 1):
            for t2 in range(1, len(token2) + 1):
                char1, char2 = token1[t1 - 1], token2[t2 - 1]
                if char1 == char2:
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    key_dist = self.keyboard_distances.get(char1, {}).get(char2, 99)
                    cost_replace = 0.8 if key_dist == 1 else 0.9 if key_dist == 2 else 1.0
                    distances[t1][t2] = min(
                        distances[t1 - 1][t2] + self.cost_delete,  # xóa
                        distances[t1][t2 - 1] + self.cost_insert,  # chèn
                        distances[t1 - 1][t2 - 1] + cost_replace,  # thay thế
                    )
        return distances[len(token1)][len(token2)]

    def suggest_corrections(self, token: str) -> List[str]:
        """Gợi ý từ sửa lỗi dựa trên khoảng cách Levenshtein có trọng số bàn phím."""
        token_lower = token.lower()
        if token_lower in self.vocabulary:
            return [token]

        candidates: List[Tuple[str, float]] = []
        for word in self.vocabulary:
            if abs(len(token_lower) - len(word)) <= self.max_distance:
                dist = self.keyboard_levenshtein_distance(token_lower, word)
                if dist <= self.max_distance:
                    candidates.append((word, dist))

        candidates.sort(key=lambda x: x[1])
        return [word for word, _ in candidates[: self.max_suggestions]]


class KeyboardGraph:
    """Tạo và quản lý ma trận khoảng cách bàn phím."""

    @staticmethod
    def build_keyboard_graph() -> Dict[str, Set[str]]:
        """Tạo đồ thị bàn phím QWERTY."""
        layout = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        graph = {char: set() for row in layout for char in row}

        for r_idx, row in enumerate(layout):
            for c_idx, char in enumerate(row):
                if r_idx > 0 and c_idx < len(layout[r_idx - 1]):
                    neighbor = layout[r_idx - 1][c_idx]
                    graph[char].add(neighbor)
                    graph[neighbor].add(char)
                if c_idx > 0:
                    neighbor = row[c_idx - 1]
                    graph[char].add(neighbor)
                    graph[neighbor].add(char)
        return graph

    @staticmethod
    def bfs_from_start_node(graph: Dict[str, Set[str]], start_node: str) -> Dict[str, int]:
        """Chạy BFS để tìm khoảng cách từ một nút bắt đầu."""
        distances = {node: -1 for node in graph}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            for neighbor in sorted(graph[current_node]):
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    @staticmethod
    def generate_distance_matrix() -> Dict[str, Dict[str, int]]:
        """Tạo ma trận khoảng cách cho bàn phím QWERTY."""
        graph = KeyboardGraph.build_keyboard_graph()
        all_chars = sorted(graph.keys())
        distance_matrix = {
            char: KeyboardGraph.bfs_from_start_node(graph, char) for char in all_chars
        }
        return distance_matrix

    @staticmethod
    def save_distance_matrix_to_csv(
        distance_matrix: Dict[str, Dict[str, int]], filename: str = "keyboard_distances.csv"
    ) -> None:
        """Lưu ma trận khoảng cách vào file CSV."""
        df = pd.DataFrame(distance_matrix).reindex(sorted(distance_matrix.keys()))
        df.to_csv(filename, index_label="char")
        print(f"Ma trận khoảng cách đã được lưu vào {filename}")

    @staticmethod
    def load_distances_from_csv(filename: str = "keyboard_distances.csv") -> Dict[str, Dict[str, int]]:
        """Tải khoảng cách bàn phím từ file CSV."""
        distances = {}
        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)[1:]
            for row in reader:
                char1 = row[0]
                char1_distances = {char2: int(dist) for char2, dist in zip(header, row[1:])}
                distances[char1] = char1_distances
        print(f"Khoảng cách bàn phím đã được tải từ {filename}")
        return distances


def load_english_vocabulary(name: str = "en-basic") -> Set[str]:
    """Tải từ vựng tiếng Anh từ NLTK."""
    vocabulary = set(words.words(name))
    print(f"Từ vựng đã được tải với {len(vocabulary)} từ.")
    return vocabulary


def load_misspell_pairs(path: str = "./missp.dat.txt") -> List[Tuple[str, str]]:
    """Tải cặp từ sai và từ đúng từ file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pairs = []  # (misspell, correct)
    current_correct = None
    for token in text.split():
        if token.startswith("$"):
            current_correct = token[1:]
        else:
            pairs.append((token.lower(), current_correct.lower()))
    return pairs


def evaluate(pairs: List[Tuple[str, str]], suggester, topk: int, vocabulary: Set[str]) -> float:
    """Đánh giá độ chính xác của gợi ý từ cho top-k."""
    hits = 0
    for mis, cor in pairs:
        desc = f"Đánh giá Top-{topk}"
        suggestions = suggester(mis, vocabulary, max_suggestions=topk, max_distance=2)
        if cor in suggestions[:topk]:
            hits += 1
    return hits / len(pairs)


def main() -> None:
    """Hàm chính để tạo ma trận khoảng cách và so sánh hai cách thực hiện."""
    # Tải dữ liệu
    vocabulary = load_english_vocabulary()
    pairs = load_misspell_pairs()
    distance_matrix = KeyboardGraph.generate_distance_matrix()
    KeyboardGraph.save_distance_matrix_to_csv(distance_matrix)
    keyboard_distances = KeyboardGraph.load_distances_from_csv()

    # Khởi tạo hai bộ sửa lỗi
    lev_corrector = LevenshteinCorrector(vocabulary)
    kb_corrector = KeyboardLevenshteinCorrector(vocabulary, keyboard_distances)

if __name__ == "__main__":
    main()