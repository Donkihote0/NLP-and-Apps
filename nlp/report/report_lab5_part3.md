# Nguyễn Đức Đạt - 23000109

## Lab 5: Xây dựng mô hình RNN cho bài toán Part-of-Speech Tagging

### Mục tiêu

Trong bài thực hành này, chúng ta sẽ áp dụng các kiến thức lý thuyết về Mạng Nơ-ron
Hồi quy (RNN) đã học để xây dựng một mô hình hoàn chỉnh cho bài toán Part-of-Speech
(POS) Tagging.
Sau khi hoàn thành bài lab, bạn có thể:
• Tải và tiền xử lý dữ liệu văn bản từ định dạng CoNLL-U.
• Xây dựng từ điển (vocabulary) cho từ và nhãn.
• Tạo một lớp Dataset tùy chỉnh trong PyTorch.
• Xây dựng một mô hình RNN đơn giản từ các khối nn.Embedding, nn.RNN, và
nn.Linear.
• Huấn luyện và đánh giá hiệu năng của mô hình trên một bộ dữ liệu thực tế.

### Các bước thực hiện

Task 1: Tải và Tiền xử lý Dữ liệu

    1.  Viết hàm đọc file .conllu:
        • Viết một hàm load_conllu(file_path) để đọc dữ liệu từ các file en_ewt-ud-train.conllu
        và en_ewt-ud-dev.conllu.
        • Hàm này cần trả về một danh sách các câu. Mỗi câu là một danh sách các
        cặp (word, upos_tag).
        • Ví dụ: [[('From', 'ADP'), ('the', 'DET'), ...], [('Another', 'DET'),
            ('sentence', 'NOUN'), ...]]
    2. Xây dựng Từ điển (Vocabulary):
        • Từ dữ liệu huấn luyện (train), tạo ra hai từ điển:
            – word_to_ix: Ánh xạ mỗi từ duy nhất sang một chỉ số (index) nguyên. Thêm
            một token đặc biệt là <UNK> cho các từ không có trong từ điển.
            – tag_to_ix: Ánh xạ mỗi nhãn UPOS duy nhất sang một chỉ số nguyên.
        • In ra kích thước của hai từ điển này.

Task 2: Tạo PyTorch Dataset và DataLoader

    1. Tạo lớp POSDataset:
        • Tạo một lớp kế thừa từ torch.utils.data.Dataset.
        • **init**: Nhận vào danh sách các câu đã xử lý và hai từ điển word_to_ix, tag_to_ix.
        • **len**: Trả về tổng số câu trong bộ dữ liệu.
        • **getitem**: Nhận vào một index và trả về một cặp tensor: (sentence_indices, tag_indices). Tensor này chứa các chỉ số của từ/nhãn trong câu tương ứng.
    2. Tạo DataLoader:
        • Khởi tạo DataLoader cho cả tập train và dev.
        • Lưu ý quan trọng: Các câu trong một batch có độ dài khác nhau. Bạn cần viết một hàm collate_fn để đệm (pad) các câu và nhãn trong cùng
        một batch về cùng một độ dài (độ dài của câu dài nhất batch đó). Sử dụng torch.nn.utils.rnn.pad_sequence với batch_first=True.

Task 3: Xây dựng Mô hình RNN

    1. Dựa trên đoạn code khái niệm trong bài giảng, hãy hoàn thiện lớp SimpleRNNForTokenClassification.
    2. Mô hình sẽ bao gồm 3 lớp chính:
        • nn.Embedding: Chuyển đổi chỉ số của từ thành vector.
        • nn.RNN: Xử lý chuỗi vector embedding.
        • nn.Linear: Ánh xạ output của RNN sang không gian nhãn để dự đoán.
    3. Hãy chú ý đến kích thước (dimension) của các tensor ở đầu vào và đầu ra của mỗi lớp.

Task 4: Huấn luyện Mô hình

    1. Khởi tạo:
        • Khởi tạo mô hình, optimizer (ví dụ: torch.optim.Adam), và loss function.
        • Sử dụng nn.CrossEntropyLoss cho bài toán này. Lưu ý rằng CrossEntropyLoss
        yêu cầu đầu vào là raw scores (logits) và bỏ qua các giá trị đệm (padding)
        khi tính loss. Bạn có thể đặt ignore_index cho giá trị padding của nhãn.
    2. Viết vòng lặp huấn luyện:
        • Lặp qua một số lượng epoch nhất định.
        • Trong mỗi epoch, lặp qua từng batch từ DataLoader huấn luyện.
        • Thực hiện 5 bước kinh điển: (1) Xóa gradient cũ, (2) Forward pass, (3) Tính
        loss, (4) Backward pass (lan truyền ngược), (5) Cập nhật trọng số.
        • In ra giá trị loss trung bình sau mỗi epoch hoặc sau một số lượng batch nhất
        định.

Task 5: Đánh giá Mô hình

    1. Viết hàm evaluate:
        • Đặt mô hình ở chế độ đánh giá: model.eval().
        • Tắt việc tính toán gradient: with torch.no_grad(): ...
        • Lặp qua từng batch trong DataLoader của tập dev.
        • Với mỗi batch, lấy dự đoán của mô hình bằng cách áp dụng torch.argmax trên
        chiều cuối cùng của output.
        • So sánh dự đoán với nhãn thật để tính toán độ chính xác (accuracy). Lưu ý:
        chỉ tính accuracy trên các token không phải là padding.
    2. Báo cáo kết quả:
        • Show độ chính xác trên tập train và dev sau mỗi epoch huấn luyện. Lựa chọn
        mô hình tốt nhất dựa trên độ chính xác trên tập dev.
        • In ra độ chính xác cuối cùng trên tập dev.
        • (Nâng cao) Viết một hàm predict*sentence(sentence) nhận vào một câu mới
        (dạng chuỗi), xử lý nó và in ra các cặp (từ, nhãn_dự*đoán).

### Cách chạy chương trình: Mở lab5_rnn_for_pos_tagging.py và chạy: `nlp/test/lab5_rnn_for_pos_tagging.py`

### Kết quả

```
Kích thước word_to_ix: 19674
Kích thước tag_to_ix: 17
Epoch [1/5] - Loss: 0.9926
Epoch 1/5
  Loss: 0.9926
  Train Acc: 0.7881
  Dev Acc:   0.7690
Epoch [2/5] - Loss: 0.5745
Epoch 2/5
  Loss: 0.5745
  Train Acc: 0.8532
  Dev Acc:   0.8178
Epoch [3/5] - Loss: 0.4249
Epoch 3/5
  Loss: 0.4249
  Train Acc: 0.8907
  Dev Acc:   0.8440
Epoch [4/5] - Loss: 0.3286
Epoch 4/5
  Loss: 0.3286
  Train Acc: 0.9150
  Dev Acc:   0.8526
Epoch [5/5] - Loss: 0.2601
Epoch 5/5
  Loss: 0.2601
  Train Acc: 0.9353
  Dev Acc:   0.8663

Best Dev Accuracy: 0.8663

Prediction:
I               → PRON
will            → AUX
show            → VERB
you             → PRON
the             → DET
true            → ADJ
power           → NOUN
of              → ADP
Sharingan       → PROPN
```

### Giải thích kết quả

```
Kích thước word_to_ix: 19,674
```

- Đây là số lượng từ độc nhất (unique tokens) được trích ra từ bộ dữ liệu train (không tính dev).
- có token <PAD>, <UNK>
- Còn lại là 19,672 từ xuất hiện trong tập huấn luyện.
- Với tập UD EWT, kích thước từ vựng khoảng 18k–20k từ là hoàn toàn bình thường.
  → Mô hình được train trên một lượng từ vựng khá lớn → có thể học được ngữ pháp tiếng Anh khá tốt.

```
Kích thước tag_to_ix: 17
```

- số lượng nhãn UPOS theo chuẩn Universal Dependencies: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ,
  NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X
  → Mô hình dự đoán đúng không gian nhãn chuẩn của POS tagging.

```
Epoch [1/5] - Loss: 0.9926
Epoch 1/5
  Loss: 0.9926
  Train Acc: 0.7881
  Dev Acc:   0.7690
Epoch [2/5] - Loss: 0.5745
Epoch 2/5
  Loss: 0.5745
  Train Acc: 0.8532
  Dev Acc:   0.8178
Epoch [3/5] - Loss: 0.4249
Epoch 3/5
  Loss: 0.4249
  Train Acc: 0.8907
  Dev Acc:   0.8440
Epoch [4/5] - Loss: 0.3286
Epoch 4/5
  Loss: 0.3286
  Train Acc: 0.9150
  Dev Acc:   0.8526
Epoch [5/5] - Loss: 0.2601
Epoch 5/5
  Loss: 0.2601
  Train Acc: 0.9353
  Dev Acc:   0.8663
```

- Mô hình đang học tốt, không bị stuck hoặc diverge.
- Loss giảm đều → tín hiệu rất tốt.
- Sau epoch 5 bạn có Loss khá nhỏ (≈ 0.26) → mô hình fit khá ổn.
- Train Accuracy
  Tăng đều và cao dần:
  Epoch 1 → 78.81%
  Epoch 5 → 93.53%
- Dev Accuracy
  Epoch 1 → 76.90%
  Epoch 5 → 86.63%
- Ý nghĩa:
  Train Acc luôn cao hơn Dev Acc → đúng bản chất, vì mô hình học tốt hơn trên dữ liệu mà nó đã thấy.
  Từ Epoch 1 → Epoch 3, cả Train và Dev đều tăng mạnh → mô hình đang học ngữ pháp hiệu quả.
  Từ Epoch 3 → 5, tốc độ tăng chậm lại → mô hình tiến gần đến mức mà Simple RNN có thể đạt.
- Dev Acc = 86.63% là rất tốt đối với một mô hình cơ bản:
  Đây là Simple RNN không có LSTM, không Bi-RNN, không CRF
  Không có pre-trained embeddings
  Batch size nhỏ
  Chỉ embedding 128 và hidden 256
  → Kết quả cực kỳ hợp lý, thậm chí là tốt.

```
Best Dev Accuracy: 0.8663
```

- lưu mô hình tốt nhất dựa trên dev accuracy.
  0.8663 = 86.63%
- Đây chính là độ chính xác state-of-the-art… đối với một mô hình RNN thuần.
  → mô hình ít bị overfit nhất.

```
Prediction:
I               → PRON
will            → AUX
show            → VERB
you             → PRON
the             → DET
true            → ADJ
power           → NOUN
of              → ADP
Sharingan       → PROPN
```

- Mô hình dự đoán hoàn toàn chính xác
- Ngữ pháp câu đúng
- Từ “Sharingan” → PROPN (tên riêng) → quá tuyệt!
  → Dự đoán này không có trong vocab train, nhưng RNN học ngữ cảnh nên chọn đúng.
  → Mô hình POS tagging đã học được cấu trúc ngữ pháp tiếng Anh khá tốt.
  → Có khả năng generalize tốt cho từ mới hoàn toàn.

### Khó khăn và cách khắc phục

- Quá trình huấn luyện tốn 1 chút thời gian, nhưng độ chính xác khá ổn
  -> Có thể tăng thêm epoch, hy sinh thêm chút thời gian nhưng độ chính xác có thể cao hơn

### Tài liệu và thư viện tham khảo

- Các thư viện của torch như utils.data, nn.utils.rnn, nn, optim
- Các cấu trúc dữ liệu Dataset, DataLoader của torch
- ChatGPT
- `https://www.geeksforgeeks.org/nlp/nlp-part-of-speech-default-tagging/`
- `https://ongxuanhong.wordpress.com/2016/09/02/gan-nhan-tu-loai-part-of-speech-tagging-pos/`
