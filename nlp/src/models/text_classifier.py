# Task 2

from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.representations.count_vectorizer import CountVectorizer
class TextClassifier:
    def __init__(self, vectorizer: CountVectorizer):
        """
        Khởi tạo TextClassifier với một đối tượng vectorizer.
        Args:
            vectorizer: Instance của lớp CountVectorizer (phải có phương thức fit_transform và transform).
        """
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện mô hình Logistic Regression trên tập văn bản và nhãn.
        Args:
            texts (List[str]): Danh sách các câu văn bản huấn luyện.
            labels (List[int]): Danh sách nhãn tương ứng.
        """
        # Biến đổi văn bản thành ma trận đặc trưng
        X = self.vectorizer.fit_transform(texts)

        # Khởi tạo mô hình Logistic Regression
        self._model = LogisticRegression(solver='liblinear', random_state=42)

        # Huấn luyện
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho danh sách văn bản mới.
        Args:
            texts (List[str]): Danh sách văn bản cần dự đoán.

        Returns:
            List[int]: Danh sách nhãn dự đoán.
        """
        if self._model is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước khi predict().")

        # Biến đổi văn bản thành ma trận đặc trưng
        X = self.vectorizer.transform(texts)

        # Dự đoán nhãn
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá mô hình.
        Args:
            y_true (List[int]): Nhãn thật.
            y_pred (List[int]): Nhãn dự đoán.

        Returns:
            Dict[str, float]: Từ điển chứa các chỉ số accuracy, precision, recall, f1.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return metrics
