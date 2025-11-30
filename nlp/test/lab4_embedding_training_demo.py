import os
import re
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Đọc dữ liệu thô (stream từng dòng)
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Bỏ dòng rỗng hoặc comment
            if not line or line.startswith('#'):
                continue
            # Giữ lại chữ cái, tách từ
            yield simple_preprocess(line)

# Huấn luyện mô hình Word2Vec
def train_word2vec(data_path, save_path):
    print("Reading and preprocessing data...")
    sentences = list(read_corpus(data_path))
    print(f"Total sentences: {len(sentences)}")

    print("\nTraining Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,   # tăng kích thước vector để biểu diễn tốt hơn
        window=5,          # ngữ cảnh 5 từ
        min_count=2,       # bỏ từ xuất hiện ít hơn 2 lần
        workers=4,         # sử dụng đa luồng
        sg=1,              # 1: skip-gram, 0: CBOW
        epochs=10          # tăng số vòng huấn luyện
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n Model saved to: {save_path}")
    return model

# Kiểm tra mô hình
def demonstrate_model(model):
    print("\nDemonstrating trained Word2Vec model...\n")

    # Từ tương tự
    target_word = "city"
    if target_word in model.wv:
        print(f"Top 5 words similar to '{target_word}':")
        for word, score in model.wv.most_similar(target_word, topn=5):
            print(f"  {word:<10} -> {score:.4f}")
    else:
        print(f"'{target_word}' not found in vocabulary.")

    # Phép toán ngữ nghĩa
    analogy = ("king", "man", "woman")
    print("\nAnalogy test: king - man + woman ≈ ?")
    try:
        result = model.wv.most_similar(
            positive=[analogy[0], analogy[2]], negative=[analogy[1]], topn=1
        )
        print(f"  {analogy[0]} - {analogy[1]} + {analogy[2]} ≈ {result[0][0]} (score={result[0][1]:.4f})")
    except KeyError:
        print(" One of the words is not in the vocabulary.")


# Main
if __name__ == "__main__":
    data_path = "./UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
    save_path = "results/word2vec_ewt.model"

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
    else:
        model = train_word2vec(data_path, save_path)
        demonstrate_model(model)

