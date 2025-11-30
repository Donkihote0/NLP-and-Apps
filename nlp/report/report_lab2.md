# Lab 2: Count Vectorization

## Tác giả: Nguyễn Đức Đạt - 23000109

## Giới thiệu

    - Trong Lab 2, bạn sẽ học cách biểu diễn văn bản dưới dạng **vector số** bằng cách triển khai mô hình **Bag-of-Words (BoW)** thông qua một **CountVectorizer**.
    - Thành phần này rất quan trọng khi áp dụng dữ liệu văn bản cho các mô hình học máy.
    - Lab này sẽ **tái sử dụng Tokenizer từ Lab 1**.

## Mục tiêu:

    - Hiểu cách hoạt động của mô hình Bag-of-Words.
    - Cài đặt **Vectorizer interface**.
    - Cài đặt **CountVectorizer** dùng Tokenizer để chuyển văn bản thành vector đếm.
    - Thử nghiệm trên một corpus nhỏ và quan sát kết quả.

## Các bước thực hiện

### Task 1: Vectorizer Interface

- File: `src/core/interfaces.py`
- Định nghĩa abstract class `Vectorizer` với các phương thức:

  ```python
  def fit(self, corpus: list[str]):
      """Học vocabulary từ tập văn bản"""

  def transform(self, documents: list[str]) -> list[list[int]]:
      """Chuyển danh sách văn bản thành danh sách vector đếm"""

  def fit_transform(self, corpus: list[str]) -> list[list[int]]:
      """Thực hiện fit và transform trên cùng một tập dữ liệu"""
  ```

---

### Task 2: CountVectorizer Implementation

1. Tạo file: `src/representations/count_vectorizer.py`.
2. Tạo class `CountVectorizer` kế thừa từ `Vectorizer`.
   - Constructor:
     ```python
     def __init__(self, tokenizer: Tokenizer):
         self.tokenizer = tokenizer
         self.vocabulary_ = {}
     ```
3. Cài đặt **fit**:

   - Khởi tạo tập rỗng để lưu tokens duy nhất.
   - Duyệt qua từng document, tokenize rồi thêm token vào tập.
   - Sau đó ánh xạ mỗi token vào một chỉ số trong `vocabulary_`.

4. Cài đặt **transform**:
   - Với mỗi document:
     - Khởi tạo vector 0 có độ dài bằng kích thước vocabulary.
     - Tokenize văn bản.
     - Với mỗi token có trong vocabulary, tăng giá trị tại index tương ứng.
   - Trả về danh sách các vector kết quả.

---

## Evaluation

1. Tạo file test: `test/lab2_test.py`.
2. Trong file test:
   - Khởi tạo `RegexTokenizer` (Lab 1).
   - Khởi tạo `CountVectorizer` với tokenizer trên.
   - Định nghĩa corpus mẫu:
     ```python
     corpus = [
         "I love NLP.",
         "I love programming.",
         "NLP is a subfield of AI."
     ]
     ```
   - Gọi `fit_transform` trên corpus.
   - In ra **vocabulary** và **document-term matrix** (ma trận vector).
3. Bonus: Chạy dữ liệu test UD-English_EWT với các bước như trên, việc đọc dữ liệu tương tự lab1

---

---

## Cấu trúc thư mục

```
root/
│
├── report/                      # Thư mục chứa báo cáo markdown cho từng lab
│   ├── report_lab1.md
│   └── report_lab2.md
│
├── src/                         # Thư mục chứa code nguồn chính
│   ├── core/                    # Thành phần cốt lõi
│   │   ├── dataset_loaders.py   # Module load dữ liệu
│   │   ├── interfaces.py        # Định nghĩa interface, abstract class
│   │   └── __init__.py
│   │
│   ├── preprocessing/           # Các bước tiền xử lý dữ liệu
│   │   ├── regex_tokenizer.py   # Tokenizer nâng cao dùng regex
│   │   ├── simple_tokenizer.py  # Tokenizer cơ bản
│   │   └── __init__.py
│   │
│   ├── representations/         # Biểu diễn dữ liệu (vectorization, embedding,…)
│   │   ├── count_vectorizer.py  # Vector hóa bằng phương pháp Bag-of-Words
│   │   └── __init__.py
│   │
│   └── __pycache__/             # File cache do Python sinh ra
│
├── test/                        # Thư mục chứa test
│   ├── lab1_test.py             # Test cho Lab 1 (tokenizer,…)
│   ├── lab2_test.py             # Test cho Lab 2 (vectorizer,…)
│   └── __init__.py
│
└── UD_English-EWT               # Thư mục chứa dữ liệu test

```

---

## Ví dụ chạy

```terminal
python -m test.lab1_test
```

Output (ví dụ rút gọn):

```
--- CountVectorizer Test ---
Vocabulary (word -> index):
.: 0
AI: 1
I: 2
NLP: 3
a: 4
is: 5
love: 6
of: 7
programming: 8
subfield: 9

Document-Term Matrix:
Doc 1: [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
Doc 2: [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
Doc 3: [1, 1, 0, 1, 1, 1, 0, 1, 0, 1]

CountVectorizer Test:
Vocabulary (word -> index):
-: 0
:: 1
A: 2
S: 3
Z: 4
a: 5
b: 6
c: 7
d: 8
e: 9
f: 10
h: 11
i: 12
k: 13
l: 14
m: 15
n: 16
o: 17
r: 18
s: 19
u: 20

Document-Term Matrix:
Doc 1: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Doc 3: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 6: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
Doc 7: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
Doc 9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 10: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 12: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
Doc 14: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
Doc 16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 17: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 18: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 19: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
Doc 20: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 21: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 22: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
Doc 23: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
Doc 24: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 25: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 26: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
Doc 27: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 28: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Doc 29: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 30: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Doc 31: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Doc 32: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 33: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 34: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 35: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 36: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 37: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 38: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 39: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Doc 40: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 41: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 42: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 43: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 44: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 45: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
Doc 46: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Doc 47: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Doc 48: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 49: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Doc 50: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## 1 số khó khăn mắc phải:

- Link dữ liệu test bị lỗi, nên phải tải data về và đọc từ file
- Các thư mục đều có file **init**.py vì nếu không, khi chạy test sẽ không import được các file khác, do nó không coi các thư mục khác là package
- Khi chạy dữ liệu test UD-English-EWT, do dữ liệu không phải list các chuỗi nên chạy sẽ lỗi, và em đang không biết fix

## 1 số thư viện, model, promt được dùng:

- List, dict từ typing
- os

## 1 số nguồn tham khảo: ChatGPT, thuật toán tách từ cơ bản từ Lập trình cơ bản
