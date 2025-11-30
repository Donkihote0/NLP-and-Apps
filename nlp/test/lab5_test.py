import sys
import os
import re
# Đảm bảo có thể import từ thư mục src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from src.representations.count_vectorizer import CountVectorizer
from src.models.text_classifier import TextClassifier
from src.preprocessing.regex_tokenizer import RegexTokenizer
# Task 1 và 3
#  Tạo dữ liệu mẫu
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

#  Khởi tạo Tokenizer và Vectorizer
tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer=tokenizer)

# hàm loại bỏ nhiễu (Task4)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # loại bỏ URL
    text = re.sub(r"<.*?>", "", text)    # loại bỏ thẻ HTML
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # bỏ ký tự đặc biệt, giữ lại chữ cái và khoảng trắng
    text = text.lower().strip()
    return text

X_train = [clean_text(t) for t in X_train]
X_test = [clean_text(t) for t in X_test]

# Khởi tạo và huấn luyện TextClassifier 
classifier = TextClassifier(vectorizer)
classifier.fit(X_train, y_train)

# Dự đoán trên tập test 
y_pred = classifier.predict(X_test)

# Đánh giá và in kết quả 
metrics = classifier.evaluate(y_test, y_pred)

print(" Evaluation Results")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")