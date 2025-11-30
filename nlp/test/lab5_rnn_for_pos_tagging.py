import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Task 1: Tải và Tiền xử lý Dữ liệu
# Hàm đọc file
def load_conllu(file_path):
    sentences = []
    current_sentence = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Câu kết thúc khi gặp dòng trống
            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            # Bỏ các dòng bắt đầu bằng "#"
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 10:
                continue  # không phải token chuẩn
            token_id = parts[0]
            if "-" in token_id or "." in token_id:
                continue  # bỏ các token multiword như 2-3, 5.1
            word = parts[1]
            upos = parts[3]
            current_sentence.append((word, upos))
    # Thêm câu cuối nếu không kết thúc bằng dòng trống
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

# Xây dựng từ điển
def build_vocab(sentences):
    word_to_ix = {"<UNK>": 0}   # token đặc biệt
    tag_to_ix = {}
    # duyệt toàn bộ các câu để lấy từ và nhãn
    for sent in sentences:
        for word, tag in sent:
            # vocab của word
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            # vocab của tag
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return word_to_ix, tag_to_ix

# Đọc dữ liệu train và dev
train_path = "nlp/data/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.conllu"
dev_path   = "nlp/data/UD_English-EWT/UD_English-EWT/en_ewt-ud-dev.conllu"
train_sentences = load_conllu(train_path)
dev_sentences   = load_conllu(dev_path)
# Xây dựng từ điển từ train
word_to_ix, tag_to_ix = build_vocab(train_sentences)
print("Kích thước word_to_ix:", len(word_to_ix))
print("Kích thước tag_to_ix:", len(tag_to_ix))


# Task 2: Tạo PyTorch Dataset và DataLoader
# Tạo lớp POSDataset
class POSDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        """
        sentences: danh sách các câu dạng [(word, tag), ...]
        word_to_ix: dict ánh xạ từ → index
        tag_to_ix: dict ánh xạ tag → index
        """
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_indices = []
        tag_indices = []
        for word, tag in sentence:
            word_idx = self.word_to_ix.get(word, self.word_to_ix["<UNK>"])
            tag_idx = self.tag_to_ix[tag]
            word_indices.append(word_idx)
            tag_indices.append(tag_idx)
        return torch.tensor(word_indices, dtype=torch.long), \
               torch.tensor(tag_indices, dtype=torch.long)

# hàm collate
def collate_fn(batch):
    """
    batch: list of (sentence_tensor, tag_tensor)
    """
    sentences = [item[0] for item in batch]
    tags = [item[1] for item in batch]
    # Pad sentences → dùng PAD = 0
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    # Pad tags → dùng padding_value = -100 để phục vụ CrossEntropyLoss
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=-100)
    return padded_sentences, padded_tags

# Tạo DataLoader
train_dataset = POSDataset(train_sentences, word_to_ix, tag_to_ix)
dev_dataset   = POSDataset(dev_sentences, word_to_ix, tag_to_ix)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


# Task 3: Xây dựng Mô hình RNN
# tạo lớp SimpleRNNForTokenClassification
class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 2. RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        # 3. Linear classifier
        self.classifier = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        """
        x: tensor shape (batch_size, seq_len)
        Trả về: logits (batch_size, seq_len, num_tags)
        """
        # 1. Embedding
        emb = self.embedding(x)
        # emb shape: (batch, seq_len, embedding_dim)
        # 2. RNN
        rnn_out, hidden = self.rnn(emb)
        # rnn_out shape: (batch, seq_len, hidden_dim)
        # 3. Linear classifier
        logits = self.classifier(rnn_out)
        # logits shape: (batch, seq_len, num_tags)
        return logits


# Task 5: Đánh giá mô hình
# hàm đánh giá (Viết trước để task 4 huấn luyện và dùng luôn)
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_sentences, batch_tags in dataloader:
            batch_sentences = batch_sentences.to(device)
            batch_tags = batch_tags.to(device)
            logits = model(batch_sentences)
            predictions = torch.argmax(logits, dim=-1)   # (B, L)
            # mask: chỉ giữ token hợp lệ (khác -100)
            mask = (batch_tags != -100)
            correct += ((predictions == batch_tags) & mask).sum().item()
            total += mask.sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy


# Task 4: Huấn luyện mô hình
# Khởi tạo, tối ưu tổn thất
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleRNNForTokenClassification(
    vocab_size=len(word_to_ix),
    embedding_dim=128,
    hidden_dim=256,
    num_tags=len(tag_to_ix)
).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) # dùng tối ưu Adam với hệ số học là 0.001
# ignore_index = -100 vì trong collate_fn ta pad nhãn bằng -100 trong hàm tổn thất
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Huấn luyện
num_epochs = 5
best_dev_acc = 0 # khởi tạo accuracy = 0
best_model_state = None # khởi tạo model tốt nhất là rỗng
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_sentences, batch_tags in train_loader:
        # Đưa dữ liệu lên GPU
        batch_sentences = batch_sentences.to(device)   # (B, L)
        batch_tags = batch_tags.to(device)             # (B, L)
        # Xóa gradient cũ
        optimizer.zero_grad()
        # Forward pass
        logits = model(batch_sentences)    # (B, L, num_tags)
        # Tính loss
        # reshape về (B*L, num_tags) và (B*L)
        loss = criterion(
            logits.view(-1, logits.shape[-1]),
            batch_tags.view(-1)
        )
        # Backward pass
        loss.backward()
        # Cập nhật w
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    # Đánh giá
    train_acc = evaluate(model, train_loader, device)
    dev_acc = evaluate(model, dev_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Train Acc: {train_acc:.4f}")
    print(f"  Dev Acc:   {dev_acc:.4f}")
    # Lưu mô hình tốt nhất
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_model_state = model.state_dict()

# Task 5 (Tiếp)
# Chạy lại model tốt nhất
model.load_state_dict(best_model_state)
print(f"\nBest Dev Accuracy: {best_dev_acc:.4f}")

# hàm dự đoán câu mới
def predict_sentence(sentence, model, word_to_ix, ix_to_tag, device):
    model.eval()
    # Tokenize đơn giản theo khoảng trắng
    words = sentence.split()
    # chuyển sang chỉ số
    indices = [
        word_to_ix.get(w, word_to_ix["<UNK>"])
        for w in words
    ]
    tensor_input = torch.tensor([indices], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor_input)
        predictions = torch.argmax(logits, dim=-1).squeeze(0)
    tags = [ix_to_tag[idx.item()] for idx in predictions]
    print("\nPrediction:")
    for w, t in zip(words, tags):
        print(f"{w:15s} → {t}")

ix_to_tag = {v: k for k, v in tag_to_ix.items()} # index dùng cho dự đoán
predict_sentence("I will show you the true power of Sharingan", model, word_to_ix, ix_to_tag, device)


