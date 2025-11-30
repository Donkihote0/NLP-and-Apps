# Lab 17: NLP Pipeline with Spark (Updated)

## Nguyễn Đức Đạt - 23000109

---

### Nội dung chính:

- Updated: Định nghĩa biến limit là số văn bản cần lấy
- Đọc file `c4-train.00000-of-01024-30K.json.gz` bằng
  `spark.read.json()` thành công, load được 2000 rows.

- Sử dụng `Pipeline` từ `org.apache.spark.ml`.

- Sử dụng `RegexTokenizer` để tách từ từ cột văn bản.

- Áp dụng `StopWordsRemover` với danh sách stopwords mặc định của
  Spark.

- Dùng `HashingTF` để chuyển tokens thành vector tần suất, sau đó
  dùng `IDF` để chuẩn hóa.
- Updated: thêm bước chuẩn hóa vecto
- Gọi `pipeline.fit(df).transform(df)` để train và transform thành công, kết quả chứa
  vector đặc trưng.

- Ghi output vào `results/lab17_pipeline_output.txt`.

- Updated: Định nghĩa độ đo cosine, sau đó lấy văn bản đầu tiên firstdoc, tìm trong dataset : top 5 văn bản có mức độ tương đầu với firstdoc

- Spark log tự động sinh ra. Ngoài ra, chương trình có thêm log:

[info] Loaded 2000 rows
[info] Stage [Load Data] finished in 3.5417772 sec
[info] Loaded 2000 rows
[info] Loaded 2000 rows
[info] Stage [Train Pipeline] finished in 2.7478427 sec
[info] Stage [Transform Data] finished in 0.1498605 sec
[info] Pipeline completed. Results saved at results/lab17_pipeline_output.txt
[info] Top 5 similar docs to: Beginners BBQ Class Taking Place in Missoula!
[info] Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class...
[info] sim=1.0000 | Beginners BBQ Class Taking Place in Missoula!
[info] Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class
[info] sim=0.1282 | If you are interested in knowing more about New Life Community Church or becoming a member, we invite you to join us April 15 at 10:30am in the Meeting Room for our Starting Points class. This class w
[info] sim=0.1263 | Adult Open Hip Hop- All 10 classes discount $195 Drop in Rate: $23 available at the front desk.
[info] She has taught the French Language continuously for decades and has been in Seattle teaching French for
[info] sim=0.1227 | Mercedes has lifted the veil and presented the new X-Class. Mercedes X-Class is the company∩┐╜s first large scale production luxury pickup and we reveal all its secrets, including the prices for the Ger
[info] sim=0.0924 | A letterman, in U.S. activities/sports, is a high school or college student who has met a specified, Although sometimes, the colors of the jacket may be customized to a certain extent by the student.

---

### Các bước trong chương trình

1.  Đặt file dữ liệu vào thư mục `data/`.
2.  Đọc dữ liệu bằng `spark.read.json`.
3.  Tạo `RegexTokenizer` để tách tokens.
4.  Dùng `StopWordsRemover` để loại bỏ stop words.
5.  Dùng `HashingTF` và `IDF` để vector hóa dữ liệu.
6.  Tạo `Pipeline` gồm các bước trên.
7.  Fit & transform dữ liệu.
8.  Ghi kết quả ra file `results/lab17_pipeline_output.txt` và log vào file log.

9.  Tạo phuowng thức tính độ đo cosine để tính độ tương đồng giữa 2 vecto
10. Test với văn bản đầu tiên, tìm top 5 văn bản có độ đo tương đồng nhất, kèm độ đo

### Cấu trúc thư mục

```
spark_lab17/
├── data/ # chứa dữ liệu đầu vào (vd: C4 dataset)
│ └── c4-train.00000-of-01024-30K.json.gz
│
├── log/ # thư mục chứa log
│ └── lab17_log.txt
│
├── project/ # file cấu hình của sbt project
│ ├── target/
│ └── build.properties
│
├── results/ # kết quả chạy pipeline
│ └── lab17_pipeline_output.txt
│
├── src/
│ └── main/
│   └── scala/
│     └── com/
│       └── donkihote/
│         └── spark/
│           └── Lab17_NLPPipeline.scala
│
├── target/ # thư mục build output tự động tạo bởi sbt
│ ├── bg-jobs/
│ ├── global-logging/
│ ├── scala-2.12/
│ ├── streams/
│ └── task-temp-directory/
│
├── build.sbt # file cấu hình chính cho project
└── report_lab17.md # file báo cáo bài lab
```

### Cách chạy code: tại thư mục root

```bash
cd spark_labs
sbt clean compile
sbt run
```

→ Log chạy sẽ xuất hiện trong console. Kết quả được lưu trong
`results/lab17_pipeline_output.txt`.

### Giải thích

- Spark load được 2000 dòng dữ liệu từ C4.\
- Sau khi pipeline chạy, mỗi văn bản được token hóa, loại bỏ
  stopwords, chuyển thành vector TF-IDF.\
- Kết quả cuối cùng là ma trận đặc trưng TF-IDF được chuẩn hóa của dữ liệu, dùng
  cho các bước học máy tiếp theo.
- Lấy văn bản đầu tiên ra (firstDoc), tìm 5 văn bản trong danh sách có sự tương đồng gần nhất thông qua độ đo cosine, nếu độ đo càng gần 1 thì 2 văn bản càng giống nhau, ngược lại thì không

### 1 số khó khăn

- **Vấn đề 1:** Cảnh báo `winutils.exe` trên Windows.\
  → Đây chỉ là cảnh báo, bỏ qua được khi chạy Spark
  standalone.\
- **Vấn đề 2:** Java version cao (17+).Do máy e là java 24 nên sẽ bị lỗi \
  → Thêm `--add-opens` vào `build.sbt` để tránh lỗi
  IllegalAccess.
- **Vấn đề 3:** Github không cho upload file data lên vì file data có dung lượng > 25mb
  → Khi clone chương trình về để chạy, cần lấy file data trên google classroom, ở thư mục spark_lab17, tạo thư mục data rồi để file .json.gz kia vào

### Nguồn tham khảo

- Tài liệu Spark ML:
  https://spark.apache.org/docs/3.5.1/ml-guide.html\
- Spark trên Windows (winutils.exe):
  https://wiki.apache.org/hadoop/WindowsProblems
- ChatGPT

### Mô hình tiền huấn luyện

       → Không sử dụng pre-trained models trong lab này.

---
