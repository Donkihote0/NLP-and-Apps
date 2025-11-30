# test/lab4_visualize_embeddings.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.downloader import load
import numpy as np
# 1. Tải model pre-trained GloVe
model = load('glove-wiki-gigaword-50')

# 2. Chọn một nhóm từ để hiển thị
words = [
    'king', 'queen', 'man', 'woman', 'prince', 'princess', 'royal',
    'computer', 'software', 'hardware', 'technology', 'internet',
    'dog', 'cat', 'wolf', 'lion', 'tiger'
]

# 3. Lấy vector tương ứng
vectors = [model[word] for word in words if word in model.key_to_index]

# 4. Giảm chiều xuống 2 chiều (chọn PCA hoặc t-SNE)

# PCA (nhanh hơn)
# reducer = PCA(n_components=2)

# t-SNE (đẹp hơn, tốn thời gian hơn)
reducer = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=3000)

vectors = np.array(vectors)
reduced_vectors = reducer.fit_transform(vectors)

# 5. Vẽ scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color='skyblue')

# 6. Gắn nhãn cho từng từ
for i, word in enumerate(words):
    if word in model.key_to_index:
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plt.title("Visualization of Word Embeddings (PCA/t-SNE)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()

