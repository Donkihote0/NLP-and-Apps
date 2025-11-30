# Nguyễn Đức Đạt - 23000109

## Lab 5: Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER)

### Mục tiêu

Trong bài thực hành này, chúng ta sẽ tiếp tục áp dụng Mạng Nơ-ron Hồi quy (RNN)
để xây dựng một mô hình hoàn chỉnh cho bài toán Nhận dạng Thực thể Tên (Named
Entity Recognition - NER).
Sau khi hoàn thành bài lab, bạn có thể:
• Tải và tiền xử lý dữ liệu NER từ thư viện datasets của Hugging Face.
• Xây dựng từ điển (vocabulary) cho từ và nhãn NER.
• Tạo một lớp Dataset tùy chỉnh trong PyTorch cho bài toán token classification.
• Xây dựng một mô hình RNN đơn giản sử dụng nn.Embedding, nn.RNN, và nn.Linear.
• Huấn luyện và đánh giá hiệu năng của mô hình trên bộ dữ liệu CoNLL 2003.

### Các bước tiến hành

- Task 1: Tải và Tiền xử lý Dữ liệu

  1. Tải dữ liệu từ Hugging Face:
     • Sử dụng hàm datasets.load_dataset("conll2003", trust_remote_code=True)
     để tải bộ dữ liệu. Thao tác này có thể mất vài phút.
     • Dữ liệu trả về là một DatasetDict chứa các split: train, validation, và test.
  2. Trích xuất câu và nhãn:
     • Từ đối tượng dataset đã tải, hãy trích xuất các câu (danh sách token) và các
     nhãn tương ứng.
     • train_sentences = dataset["train"]["tokens"]
     • train_tags = dataset["train"]["ner_tags"]
     • Lưu ý: Nhãn ner_tags đang ở dạng số nguyên. Bạn cần lấy ánh xạ từ số sang
     tên nhãn (string) bằng cách truy cập dataset["train"].features["ner_tags"].feature.names.
     Hãy chuyển đổi tất cả các nhãn số về dạng string (ví dụ: B-PER, I-PER, O).
  3. Xây dựng Từ điển (Vocabulary):
     a. Từ dữ liệu huấn luyện (train), tạo ra hai từ điển:
     • word_to_ix: Ánh xạ mỗi từ duy nhất sang một chỉ số (index) nguyên. Thêm
     một token đặc biệt là <UNK> cho các từ không có trong từ điển và <PAD>
     cho việc đệm (padding).
     • tag_to_ix: Ánh xạ mỗi nhãn NER (dạng string) duy nhất sang một chỉ số
     nguyên.
     b. In ra kích thước của hai từ điển này.

- Task 2: Tạo PyTorch Dataset và DataLoader
  1. Tạo lớp NERDataset:
     • Tạo một lớp kế thừa từ torch.utils.data.Dataset.
     • **init**: Nhận vào danh sách các câu (dạng token), danh sách các chuỗi
     nhãn, và hai từ điển word_to_ix, tag_to_ix.
     • **len**: Trả về tổng số câu trong bộ dữ liệu.
     • **getitem**: Nhận vào một index và trả về một cặp tensor: (sentence_indices,
     tag_indices). Tensor này chứa các chỉ số của từ/nhãn trong câu tương ứng.
     Sử dụng token <UNK> cho các từ không có trong word_to_ix.
  2. Tạo DataLoader:
     • Khởi tạo DataLoader cho cả tập train và validation.
     • Viết một hàm collate_fn để đệm (pad) các câu và nhãn trong cùng một
     batch về cùng một độ dài (độ dài của câu dài nhất batch đó). Sử dụng
     torch.nn.utils.rnn.pad_sequence với batch_first=True. Giá trị padding cho câu
     nên là index của token <PAD>, và giá trị padding cho nhãn có thể là một số
     nguyên đặc biệt (ví dụ: -1 hoặc index của một nhãn <PAD> nếu bạn thêm nó
     vào tag_to_ix).
- Task 3: Xây dựng Mô hình RNN

  1. Dựa trên lớp SimpleRNNForTokenClassification đã thấy trong bài lab về POS
     tagging, hãy định nghĩa lại mô hình.
  2. Mô hình sẽ bao gồm 3 lớp chính:
     • nn.Embedding: Chuyển đổi chỉ số của từ thành vector.
     • nn.RNN (hoặc nn.LSTM, nn.GRU để có kết quả tốt hơn): Xử lý chuỗi vector embedding.
     • nn.Linear: Ánh xạ output của RNN sang không gian nhãn để dự đoán.
  3. Hãy khởi tạo mô hình với các tham số phù hợp: vocab_size, embedding_dim,
     hidden_dim, và output_size (số lượng nhãn NER).

- Task 4: Huấn luyện Mô hình
  1. Khởi tạo:
     • Khởi tạo mô hình, optimizer (ví dụ: torch.optim.Adam), và loss function.
     • Sử dụng nn.CrossEntropyLoss. Đây là lựa chọn phù hợp cho bài toán phân loại
     đa lớp trên từng token.
     • Quan trọng: Thiết lập tham số ignore_index của CrossEntropyLoss bằng với
     giá trị bạn đã dùng để đệm cho nhãn trong collate_fn. Điều này giúp loss
     function bỏ qua các vị trí padding khi tính toán.
  2. Viết vòng lặp huấn luyện:
     • Lặp qua một số lượng epoch nhất định (ví dụ: 3-5 epochs).
     • Trong mỗi epoch, lặp qua từng batch từ DataLoader huấn luyện.
     • Thực hiện 5 bước kinh điển: (1) Xóa gradient cũ, (2) Forward pass, (3) Tính
     loss, (4) Backward pass, (5) Cập nhật trọng số.
     • In ra giá trị loss trung bình sau mỗi epoch.
- Task 5: Đánh giá Mô hình
  1. Viết hàm evaluate:
     • Đặt mô hình ở chế độ đánh giá: model.eval().
     • Tắt việc tính toán gradient: with torch.no_grad(): ...
     • Lặp qua từng batch trong DataLoader của tập validation.
     • Với mỗi batch, lấy dự đoán của mô hình bằng cách áp dụng torch.argmax trên
     chiều cuối cùng của output.
     • So sánh dự đoán với nhãn thật để tính toán độ chính xác (accuracy). Lưu ý:
     chỉ tính accuracy trên các token không phải là padding.
     • (Nâng cao) Để đánh giá NER một cách chính xác hơn, người ta thường dùng
     các chỉ số như Precision, Recall, và F1-score trên từng loại thực thể. Thư viện
     seqeval có thể giúp bạn làm điều này.
  2. Báo cáo kết quả:
     • In ra độ chính xác cuối cùng trên tập validation.
     • Viết một hàm predict*sentence(sentence) nhận vào một câu mới (dạng chuỗi),
     xử lý nó và in ra các cặp (từ, nhãn_dự*đoán).

### Các bước chạy chương trình: Mở lab5_rnn_for_ner.py và chạy: `nlp/test/lab5_rnn_for_ner.py`

### Kết quả output

```
Danh sách nhãn NER: ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']

Kích thước word_to_ix: 23626
Kích thước tag_to_ix: 9
Epoch [1/5] - Loss: 0.5944
Validation Accuracy: 0.8754
Epoch [2/5] - Loss: 0.2897
Validation Accuracy: 0.9142
Epoch [3/5] - Loss: 0.1694
Validation Accuracy: 0.9250
Epoch [4/5] - Loss: 0.0982
Validation Accuracy: 0.9233
Epoch [5/5] - Loss: 0.0554
Validation Accuracy: 0.9293
[('VNU', 'B-ORG'), ('University', 'O'), ('is', 'O'), ('located', 'O'), ('in', 'O'), ('Hanoi', 'B-LOC')]
```

### Giải thích kết quả

```
Danh sách nhãn NER: ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
```

- Đây là 9 nhãn BIO của bộ dữ liệu CoNLL2003.
- B-XXX = bắt đầu một thực thể (Location, Person, Org, Misc).
- I-XXX = tiếp tục thực thể.
- O = token không thuộc thực thể.
  → Như vậy, mô hình của bạn đang giải quyết 9 lớp phân loại cho mỗi token.

```
Kích thước word_to_ix: 23626
Kích thước tag_to_ix: 9
```

- Bạn có 23626 từ khác nhau trong toàn dataset.
- Mỗi từ được ánh xạ thành chỉ số để đưa vào Embedding layer.
- tag_to_ix = 9 vì có 9 nhãn NER (như trên).
  → Đây là quy mô hợp lý cho CoNLL-2003 sau khi xử lý.

```
Epoch [1/5] - Loss: 0.5944 | Val Acc: 0.8754
Epoch [2/5] - Loss: 0.2897 | Val Acc: 0.9142
Epoch [3/5] - Loss: 0.1694 | Val Acc: 0.9250
Epoch [4/5] - Loss: 0.0982 | Val Acc: 0.9233
Epoch [5/5] - Loss: 0.0554 | Val Acc: 0.9293

```

- Epoch 1 → 2
  Loss giảm từ 0.59 → 0.28
  Accuracy tăng từ 87.5% → 91.4%
  → Đây là giai đoạn mô hình học kiến thức cơ bản rất nhanh.

- Epoch 2 → 3
  Loss giảm: 0.28 → 0.16
  Accuracy tăng: 91.4% → 92.5%
  → Mô hình bắt đầu học được các đặc trưng sâu hơn: chuỗi BIO, vị trí, cấu trúc câu.

- Epoch 3 → 4
  Loss giảm: 0.16 → 0.09
  Accuracy hơi giảm: 92.5% → 92.33%
  → Tín hiệu cho thấy mô hình bắt đầu hơi overfit:
  Loss train tiếp tục giảm, nhưng ảnh hưởng validation nhỏ.

- Epoch 4 → 5
  Loss giảm mạnh: 0.09 → 0.055
  Accuracy tăng: 92.33% → 92.93%
  → Mô hình hội tụ tốt, không overfit nặng.
  → Accuracy ~93% là mức trung bình tốt cho LSTM không-CRF.

```
[('VNU', 'B-ORG'), ('University', 'O'), ('is', 'O'), ('located', 'O'), ('in', 'O'), ('Hanoi', 'B-LOC')]
```

- "VNU" → B-ORG
  Chính xác ⇒ VNU là "Vietnam National University".
- "University" → O
  Điều này phụ thuộc vào corpus CoNLL (trong CoNLL từ này thường không được gắn nhãn ORG, trừ khi là "University of …").
  → Mô hình phản ứng đúng theo dữ liệu gốc.
- "is", "located", "in" → O
  Hợp lý (các từ không phải thực thể).
- "Hanoi" → B-LOC
  Chính xác.
  → Tổng thể: Mô hình dự đoán trúng 5/6 token hoàn hảo theo thói quen gán nhãn của CoNLL.

### Tổng kết

- Mô hình học tốt → Loss giảm mượt, Accuracy tăng đều.
- Accuracy đạt 92–93% là đúng kỳ vọng cho LSTM NER không dùng CRF.
- Dự đoán mẫu thực tế đúng logic.

### Khó khăn và cách khắc phục

- Quá trình huấn luyện tốn 1 chút thời gian
  -> Có thể tăng thêm epoch, hy sinh thêm chút thời gian nhưng độ chính xác có thể cao hơn
- Dữ liệu khá lỗi thời, không thể tải từ thư viện của python đươc
  -> Tải trực tiếp trên Google và đọc

### Tài liệu và thư viện tham khảo

- Các thư viện của torch như utils.data, nn.utils.rnn, nn, optim
- Các cấu trúc dữ liệu Dataset, DataDict của torch
- ChatGPT
- `https://www.kaggle.com/code/ritvik1909/named-entity-recognition-rnn`
