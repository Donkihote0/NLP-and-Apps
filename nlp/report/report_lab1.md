# Lab 1: Tokenization

## Tác giả: Nguyễn Đức Đạt - 23000109

## Giới thiệu: Trong Lab này, ta sẽ:

    •Tìm hiểu về khái niệm và tầm quan trọng của việc tách từ (tokenization) trong NLP.
    • Tự tay cài đặt một bộ tách từ đơn giản (SimpleTokenizer) dựa trên khoảng trắng
    và xử lý dấu câu cơ bản.
    • (Bonus) Cài đặt một bộ tách từ nâng cao hơn (RegexTokenizer) sử dụng biểu thức
    chính quy để xử lý các trường hợp phức tạp hơn.
    • Áp dụng các bộ tách từ đã cài đặt lên một phần của tập dữ liệu thực tế
    UD_English-EWT để quan sát và so sánh kết quả.

## Mục tiêu:

    •Hiểu và triển khai cơ chế tokenization.
    •So sánh hai phương pháp:
        •Simple Tokenizer: dựa trên khoảng trắng và xử lý dấu câu cơ bản.
        •Regex-based Tokenizer: sử dụng biểu thức chính quy để tách token chính xác hơn.
    •Kiểm tra tokenizer với tập dữ liệu thực tế.

## Các bước thực hiện

### Task 1: Simple Tokenizer

1. **Định nghĩa Interface**

   - File: `src/core/interfaces.py`
   - Tạo abstract class `Tokenizer` với phương thức:
     ```python
     def tokenize(self, text: str) -> list[str]:
         ...
     ```

2. **Cài đặt Simple Tokenizer**
   - File: `src/preprocessing/simple_tokenizer.py`
   - Tạo class `SimpleTokenizer` kế thừa `Tokenizer`.
   - Triển khai `tokenize` với các bước:
     - Chuyển text thành chữ thường.
     - Tách từ theo khoảng trắng
     * Tạo token rỗng
     * Duyệt qua từng kí tự, nếu là chữ cái thì thêm vào token
     * Nếu là khoảng trắng hoặc dấu thì đó là dấu hiệu kết thúc token, rồi them token đó vào list
     - Xử lý dấu câu cơ bản (.,?!).
     - Ví dụ:
       ```
       "Hello, world!" -> ["hello", ",", "world", "!"]
       ```

---

### Task 2: Regex-based Tokenizer (Bonus)

1. **Cài đặt Regex Tokenizer**
   - File: `src/preprocessing/regex_tokenizer.py`
   - Tạo class `RegexTokenizer` kế thừa `Tokenizer`.
   - Dùng regex để tách token, cụ thể là hàm có sẵn trong thư viện re là findall()
     ```regex
     \w+|[^\w\s]
     ```

---

### Task 3: Tokenization với UD_English-EWT Dataset

1. **Load dữ liệu**

   - Import hàm:
     ```python
     from src.core.dataset_loaders import load_raw_text_data
     ```
   - Đường dẫn ví dụ:
     ```python
     dataset_path = "D:/Studying/NLP/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
     raw_text = load_raw_text_data(dataset_path)
     ```
   - Thêm test vào file test tương tự các tasks trên

2. **Thử nghiệm Tokenizer**
   - Ở Terminal, tại thư mục gốc, chạy lệnh : `python -m test.lab1_test` để chạy task 1, 2 và 3

---

## Evaluation

- Kết quả sẽ in ra **tokens** từ SimpleTokenizer và RegexTokenizer để so sánh.
- Quan sát sự khác biệt giữa cách tách token thủ công và cách sử dụng regex.
  - RegexTokenizer có cấu trúc chặt chẽ hơn so với SimpleTokenizer, và không cần chuyển từ về in thường
  - SimpleTokenizer có 1 số hạn chế về việc tách các từ ghép nối bởi dấu nối
- Đặc biệt chú ý khi xử lý dấu câu, số và các trường hợp thực tế trong tập UD_English-EWT.

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
Testing SimpleTokenizer:
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
SimpleTokenizer:
In: Hello, world! This is a test.
Out: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

In: NLP is fascinating... isn't it?
Out: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', 't', 'it', '?']

In: Let's see how it handles 123 numbers and punctuation!
Out: ['let', 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

RegexTokenizer:
In: Hello, world! This is a test.
Out: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']

In: NLP is fascinating... isn't it?
Out: ['NLP', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

In: Let's see how it handles 123 numbers and punctuation!
Out: ['Let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']


--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ['al', 'zaman', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim']
RegexTokenizer Output (first 20 tokens): ['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah', 'al', '-', 'Ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

## 1 số khó khăn mắc phải:

- Link dữ liệu test bị lỗi, nên phải tải data về và đọc từ file
- Các thư mục đều có file **init**.py vì nếu không, khi chạy test sẽ không import được các file khác, do nó không coi các thư mục khác là package

## 1 số thư viện, model, promt được dùng:

- ABC và abstractmethod từ abc
- List từ typing
- re

## 1 số nguồn tham khảo: ChatGPT, thuật toán tách từ cơ bản từ Lập trình cơ bản
