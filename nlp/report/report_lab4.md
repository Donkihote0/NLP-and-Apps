# **Lab 4 Report – Word Embeddings & Distributed Training**

## 🔹 1. Giải thích các bước thực hiện

### **Task 1 – Setup**

- Cài đặt thư viện `gensim` để tải và sử dụng mô hình word embedding có sẵn.
- Cài đặt thêm `pyspark` cho phần nâng cao.
- Tạo file `requirements.txt` gồm:
  ```
  gensim
  pyspark
  ```
- Cài đặt:
  ```bash
  pip install -r requirements.txt
  ```

### **Task 2 – Word Embedding Exploration**

- Tạo file `src/representations/word_embedder.py`.
- Cài đặt lớp `WordEmbedder` gồm các hàm:
  - `get_vector(word)` → Lấy vector của từ.
  - `get_similarity(word1, word2)` → Tính độ tương đồng cosine.
  - `get_most_similar(word)` → Tìm top N từ đồng nghĩa gần nhất.
- Dùng mô hình **pre-trained** `glove-wiki-gigaword-50`.

### **Task 3 – Document Embedding**

- Viết hàm `embed_document(document)` để tính vector đại diện cho văn bản bằng cách **trung bình các vector từ** trong câu.
- Nếu câu không có từ hợp lệ (OOV), trả về vector 0.

### **Task 4 – Testing**

- Tạo file `test/lab4_test.py` để chạy thử:
  - Lấy vector của từ `"king"`.
  - Tính similarity giữa `"king"`–`"queen"` và `"king"`–`"man"`.
  - In ra 10 từ gần `"computer"`.
  - Nhúng câu `"The queen rules the country."`.

### **Bonus – Training Custom Word2Vec**

- Tạo script `test/lab4_embedding_training_demo.py`.
- Huấn luyện mô hình Word2Vec từ đầu bằng `gensim` trên dataset **EWT (English Web Treebank)**.
- Lưu mô hình vào `results/word2vec_ewt.model`.
- Kiểm tra với từ `"city"` và phép toán `"king - man + woman"`.

### **Advanced – Distributed Word2Vec with Spark**

- Cài đặt và chạy `pyspark`.
- Tạo file `test/lab4_spark_word2vec_demo.py`:
  - Đọc dữ liệu JSON (cột `"text"`).
  - Làm sạch văn bản, tokenize bằng Spark.
  - Huấn luyện mô hình Word2Vec phân tán (100 chiều).
  - Tìm 5 từ gần `"computer"`.
- Spark giúp huấn luyện nhanh hơn và xử lý dữ liệu lớn hơn RAM của máy.

---

## 🔹 2. Hướng dẫn chạy code

### 🧩 Chạy các phần:

```bash
cd nlp
# Kiểm tra các chức năng của WordEmbedder
python test/lab4_test.py

# Huấn luyện Word2Vec từ đầu
python test/lab4_embedding_training_demo.py

# Huấn luyện Word2Vec với Apache Spark
python test/lab4_spark_word2vec_demo.py

# Biểu đồ trực quan hóa
python test/lab4_visualize_embeddings.py
```

Kết quả sẽ in ra các từ tương tự, độ tương đồng giữa các cặp từ và vector văn bản trung bình.

---

## 🔹 3. Phân tích kết quả

### **a. Mô hình pre-trained (GloVe)**

```
LAB 4: Word Embedding Exploration
Loading model 'glove-wiki-gigaword-50' from gensim...
Model 'glove-wiki-gigaword-50' loaded successfully!

Vector for 'king':
[ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
  0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
 -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
 -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
  0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
 -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
 -0.51042 ]
Vector shape: (50,)

Similarity between 'king' and 'queen': 0.7839
Similarity between 'king' and 'man': 0.5309

Top 10 words similar to 'computer':
  computers    -> 0.9165
  software     -> 0.8815
  technology   -> 0.8526
  electronic   -> 0.8126
  internet     -> 0.8060
  computing    -> 0.8026
  devices      -> 0.8016
  digital      -> 0.7992
  applications -> 0.7913
  pc           -> 0.7883
Word 'The' not found in vocabulary (OOV).

Document embedding for: The queen rules the country.
Vector shape: (50,)
First 10 dimensions: [-0.0288  0.3884 -0.5892  0.0238  0.0468  0.1964 -0.3041 -0.1142 -0.0122
 -0.4695]
```

- `similarity("king", "queen")` ≈ **0.7839**
- `similarity("king", "man")` ≈ **0.5309**
  → GloVe học tốt mối quan hệ giới tính và ngữ nghĩa.
- Top từ gần `"computer"`:
  ```
  computers, software,technology, electronic, internet, computing, devices, digital,applications, pc
  ```
  → Các từ đều thuộc cùng trường nghĩa.

### **b. Mô hình tự huấn luyện (Word2Vec - EWT)**

```
Reading and preprocessing data...
Total sentences: 14225

Training Word2Vec model...

Model saved to: results/word2vec_ewt.model

Demonstrating trained Word2Vec model...

Top 5 words similar to 'city':
dance      -> 0.9297
kabul      -> 0.9266
complex    -> 0.9258
serving    -> 0.9240
established -> 0.9207

Analogy test: king - man + woman ≈ ?
king - man + woman ≈ easily (score=0.9292)
```

→ Các từ có thể cùng loại ngữ pháp, nhưng chưa chính xác ngữ nghĩa → do **tập dữ liệu nhỏ**.

- Phép toán `"king - man + woman"` cho ra `"easily"` → mô hình **chưa học được mối quan hệ ngữ nghĩa sâu**.

### **c. Huấn luyện Word2Vec với Apache Spark**

```
WARNING: Using incubator modules: jdk.incubator.vector
25/10/16 20:09:20 WARN Shell: Did not find winutils.exe: java.io.FileNotFoundException: java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset. -see https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/10/16 20:09:21 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Spark session started.

 Loading dataset from D:/Studying/NLP/c4-train.00000-of-01024-30K.json.gz ...
Loaded 30000 documents with 'text' field.

Preprocessing text...

Training Word2Vec model...
25/10/16 20:09:52 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS

Model trained successfully!

 Top 5 words similar to 'computer':
  desktop    -> 0.6994
  computers  -> 0.6979
  applets    -> 0.6236
  software   -> 0.6076
  ipads      -> 0.6065

Stopping Spark session...
Done.
```

→ Kết quả rất hợp lý → Spark phân tán giúp huấn luyện nhanh và ổn định trên dataset lớn.

### **d. Biểu đồ trực quan hóa**

```
![alt text](image.png)
```

### **e. So sánh giữa pre-trained và custom**

| Tiêu chí           | GloVe (pre-trained)        | Word2Vec (EWT)                  |
| ------------------ | -------------------------- | ------------------------------- |
| Dữ liệu huấn luyện | Wikipedia + Gigaword       | English Web Treebank (~50k câu) |
| Vector size        | 50                         | 100                             |
| Quan hệ ngữ nghĩa  | Rất rõ ràng (“king–queen”) | Yếu, nhiều noise                |
| Từ đồng nghĩa      | Chính xác                  | Bị lẫn, lệch nghĩa              |
| Tốc độ huấn luyện  | Cực nhanh (chỉ tải)        | Trung bình (vài phút)           |

**Kết luận:**  
➡️ GloVe vẫn vượt trội về độ chính xác.  
➡️ Word2Vec tự huấn luyện có thể cải thiện bằng cách **tăng dữ liệu và epochs**.

---

## 🔹 4. Biểu đồ trực quan hóa

-Phương pháp

-Để quan sát không gian ngữ nghĩa, ta sử dụng PCA hoặc t-SNE để giảm chiều dữ liệu từ 50/100 chiều xuống 2 chiều.
Điều này giúp hiển thị mối quan hệ giữa các từ một cách trực quan.

-Cách thực hiện

+Chọn một nhóm từ thuộc các chủ đề khác nhau (ví dụ: hoàng gia, công nghệ, động vật).

+Dùng mô hình GloVe để lấy vector embedding của từng từ.

+Dùng t-SNE (sklearn.manifold.TSNE) để giảm chiều xuống 2D.

+Dùng matplotlib để vẽ biểu đồ scatter plot, mỗi điểm là một từ.

---

## 🔹 5. Khó khăn & Giải pháp

| Khó khăn                                       | Giải pháp                                                     |
| ---------------------------------------------- | ------------------------------------------------------------- |
| -Lỗi `ModuleNotFoundError` do cấu trúc thư mục | Thêm file `__init__.py` và chạy từ thư mục gốc                |
| -Một số từ không tồn tại trong vocab (OOV)     | Bỏ qua từ OOV hoặc trả về vector 0                            |
| -Dữ liệu EWT quá nhỏ                           | Tăng epochs, hoặc thay bằng corpus lớn hơn (Wikipedia, text8) |
| -file lab4_test.py ko chạy giống mấy file      | - vào trong thư mục test rồi chạy                             |
| test kia                                       |                                                               |

---

## 🔹 6. Tài liệu tham khảo

- **Gensim Documentation:** [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
- **Spark MLlib Word2Vec:** [https://spark.apache.org/docs/latest/ml-features.html#word2vec](https://spark.apache.org/docs/latest/ml-features.html#word2vec)
- **GloVe Pre-trained Models:** [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- **Dataset:** Universal Dependencies English EWT — [https://universaldependencies.org/](https://universaldependencies.org/)

---
