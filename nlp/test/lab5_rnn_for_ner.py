import os
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn

# Task 1
#  Hàm đọc file CoNLL
def read_conll_file(path):
    sentences = []
    tags = []
    words = []
    ner_tags = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:   # kết thúc 1 câu
                if words:
                    sentences.append(words)
                    tags.append(ner_tags)
                    words = []
                    ner_tags = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            ner = parts[-1]  # cột cuối là NER tag
            words.append(word)
            ner_tags.append(ner)
        # câu cuối
        if words:
            sentences.append(words)
            tags.append(ner_tags)
    return sentences, tags

#  Đọc 3 file train, valid, test
train_path = "nlp/data/conll2003/eng.train"
valid_path = "nlp/data/conll2003/eng.testa"
test_path  = "nlp/data/conll2003/eng.testb"
train_tokens, train_tags = read_conll_file(train_path)
valid_tokens, valid_tags = read_conll_file(valid_path)
test_tokens, test_tags = read_conll_file(test_path)

# Tạo tập nhãn unique để ánh xạ → số
unique_tags = sorted({tag for sentence in train_tags for tag in sentence}) # lấy tập các nhãn ở tập train
tag2id = {t: i for i, t in enumerate(unique_tags)} # Tạo từ điển ánh xạ nhãn string → số nguyên.
id2tag = {i: t for t, i in tag2id.items()} # Tạo ánh xạ ngược số → nhãn string.
print("\nDanh sách nhãn NER:", unique_tags)

#  Chuyển nhãn string → số
train_tag_ids = [[tag2id[t] for t in s] for s in train_tags]
valid_tag_ids = [[tag2id[t] for t in s] for s in valid_tags]
test_tag_ids  = [[tag2id[t] for t in s] for s in test_tags]

#  Tạo DatasetDict 
dataset = DatasetDict({
    "train": Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_tag_ids}),
    "validation": Dataset.from_dict({"tokens": valid_tokens, "ner_tags": valid_tag_ids}),
    "test": Dataset.from_dict({"tokens": test_tokens, "ner_tags": test_tag_ids}),
})

# Trích xuất câu và nhãn 
train_sentences = dataset["train"]["tokens"]
train_tags = dataset["train"]["ner_tags"]
valid_sentences = dataset["validation"]["tokens"]
valid_tags = dataset["validation"]["ner_tags"]
test_sentences = dataset["test"]["tokens"]
test_tags = dataset["test"]["ner_tags"]

#  Chuyển tất cả nhãn số sang nhãn string
train_tags_str = [[id2tag[tag_id] for tag_id in sent] for sent in train_tags]
valid_tags_str = [[id2tag[tag_id] for tag_id in sent] for sent in valid_tags]
test_tags_str  = [[id2tag[tag_id] for tag_id in sent] for sent in test_tags]

#  Vocabulary cho từ (word_to_ix)
word_to_ix = {"<PAD>": 0, "<UNK>": 1}
for sentence in train_tokens:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) # thêm lần lượt vào cuối danh sách

# Vocabulary của tag (tag_to_ix)
tag_to_ix = tag2id  # đã tạo ở trên

print("\nKích thước word_to_ix:", len(word_to_ix))
print("Kích thước tag_to_ix:", len(tag_to_ix))


# Task 2
# Tọa lớp NerDataset kế thừa Dataset
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        """
        sentences: danh sách câu (mỗi câu là danh sách token)
        tags: danh sách nhãn (mỗi câu là danh sách nhãn string)
        word_to_ix: từ điển word -> index
        tag_to_ix: từ điển tag -> index
        """
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unk_id = word_to_ix.get("<UNK>") # id cho <UNK>

    def __len__(self):
        """Trả về số câu trong dataset"""
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        nhận vào 1 index
        Trả về:
            sentence_indices: tensor chứa id của từ
            tag_indices: tensor chứa id của nhãn
        """
        sentence = self.sentences[idx]
        tag_sequence = self.tags[idx]

        # Chuyển token -> index
        sentence_indices = [self.word_to_ix.get(word, self.unk_id) for word in sentence]

        # Chuyển tag string -> index
        tag_indices = [self.tag_to_ix[tag] for tag in tag_sequence]
        return (torch.tensor(sentence_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long))

# Hàm collate_fn (đệm độ dài câu và nhãn)
def collate_fn(batch):
    """
    batch: list của (sentence_tensor, tag_tensor)
    Trả về:
        padded_sentences: tensor (batch_size, max_len)
        padded_tags: tensor (batch_size, max_len)
        lengths: độ dài thực của từng câu
    """
    # Tách câu và nhãn
    sentences, tags = zip(*batch)   # tuple list
    lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)
    # Padding cho câu bằng index của <PAD>
    PAD_WORD = word_to_ix["<PAD>"]
    padded_sentences = pad_sequence(
        sentences,
        batch_first=True,
        padding_value=PAD_WORD
    )
    # Padding cho nhãn bằng -1 (không tham gia tính loss)
    padded_tags = pad_sequence(
        tags,
        batch_first=True,
        padding_value=-1
    )
    return padded_sentences, padded_tags, lengths

# Tạo DataLoader
batch_size = 32
train_dataset = NERDataset(
    sentences=train_sentences,
    tags=train_tags_str,
    word_to_ix=word_to_ix,
    tag_to_ix=tag_to_ix
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
valid_dataset = NERDataset(
    sentences=valid_sentences,
    tags=valid_tags_str,   # nhãn dạng string
    word_to_ix=word_to_ix,
    tag_to_ix=tag_to_ix
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


# Task 3: Xây dựng mô hình RNN
class NERRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        """
        vocab_size: số lượng từ trong word_to_ix
        embedding_dim: kích thước vector embedding
        hidden_dim: số chiều trạng thái ẩn của RNN/LSTM
        output_size: số lượng nhãn NER (tag_to_ix)
        """
        super(NERRNN, self).__init__()
        # 1. Lớp embedding: chuyển từ index -> vector
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0   # <PAD> có index = 0
        )
        # 2. LSTM layer (có thể thay bằng nn.RNN hoặc nn.GRU)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False    # Có thể đặt True nếu muốn BiLSTM
        )
        # 3. Linear để đưa hidden state -> số lượng nhãn
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lengths):
        """
        x: batch câu đã padding, shape (batch_size, max_len)
        lengths: độ dài thực của mỗi câu
        """
        # 1. Lấy embedding
        embedded = self.embedding(x)  # (batch_size, max_len, embedding_dim)
        # 2. Pack sequences để tối ưu RNN khi có padding
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed_emb)
        # 3. Giải nén (unpack) để đưa qua Linear
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (batch_size, max_len, hidden_dim)
        # 4. Linear để dự đoán nhãn tại mỗi thời điểm
        logits = self.fc(lstm_out)  # (batch_size, max_len, output_size)
        return logits


# Task 4: Huấn luyện mô hình
# khởi tạm hàm tối ưu và loss
import torch.optim as optim

# Tham số mô hình
vocab_size = len(word_to_ix)
output_size = len(tag_to_ix)
embedding_dim = 128
hidden_dim = 256

# Khởi tạo mô hình
model = NERRNN(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_size=output_size
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # chạy lên GPU
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3) # dùng tối ưu Adam
# ignore_index phải cùng giá trị padding nhãn trong collate_fn (ở trên dùng -1)
criterion = nn.CrossEntropyLoss(ignore_index=-1) # dùng cross entropy

# Task 5: Đánh giá mô hình
# hàm đánh giá
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for sentences, tags, lengths in data_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            # Forward
            logits = model(sentences, lengths)
            predictions = torch.argmax(logits, dim=-1)
            # Đếm accuracy (chỉ token không phải padding)
            mask = (tags != -1)  # true tại các vị trí có nghĩa
            correct = (predictions == tags) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    accuracy = total_correct / total_tokens
    return accuracy

# Task 4 (tiếp)
# Huấn luyện với epochs = 5
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_sentences, batch_tags, batch_lengths in train_loader:
        # Đưa data lên GPU
        batch_sentences = batch_sentences.to(device)
        batch_tags = batch_tags.to(device)
        batch_lengths = batch_lengths.to(device)
        # Xóa gradient
        optimizer.zero_grad()
        # Forward pass
        logits = model(batch_sentences, batch_lengths)
        # logits shape: (batch, max_len, num_tags)
        # Reshape để phù hợp CrossEntropyLoss:
        # CE expects: (batch*max_len, num_classes) vs (batch*max_len)
        logits_reshaped = logits.view(-1, output_size)
        tags_reshaped = batch_tags.view(-1)
        # Tính loss
        loss = criterion(logits_reshaped, tags_reshaped)
        # Backpropagation
        loss.backward()
        # Cập nhật w
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    val_acc = evaluate(model, valid_loader, device)
    print(f"Validation Accuracy: {val_acc:.4f}")

# Hàm dự đoán câu mới    
def predict_sentence(sentence, model, word_to_ix, idx_to_tag, device):
    model.eval()
    # Tokenize
    words = sentence.split()
    # Convert to index, OOV -> 0
    indices = [word_to_ix.get(w.lower(), 0) for w in words]
    length = torch.tensor([len(indices)])
    # Tensor batch=1
    tensor = torch.tensor([indices]).to(device)
    with torch.no_grad():
        logits = model(tensor, length.to(device))
        predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    # Print
    result = []
    for w, p in zip(words, predictions):
        result.append((w, idx_to_tag[p]))
    return result
    
print(predict_sentence(
    "VNU University is located in Hanoi",
    model, word_to_ix, id2tag, device
))
