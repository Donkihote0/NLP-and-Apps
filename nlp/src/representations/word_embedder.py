import gensim.downloader as api
import numpy as np
from src.preprocessing.regex_tokenizer import RegexTokenizer  # Import RegexTokenizer từ Lab 1
class WordEmbedder:
    def __init__(self, model_name: str):
        """
        Khởi tạo lớp WordEmbedder, tải mô hình embedding từ gensim.
        :param model_name: Tên mô hình (ví dụ: 'glove-wiki-gigaword-50')
        """
        print(f"Loading model '{model_name}' from gensim...")
        self.model = api.load(model_name)
        self.tokenizer = RegexTokenizer()
        print(f"Model '{model_name}' loaded successfully!")
    
    def get_vector(self, word: str):
        """
        Lấy vector embedding của một từ.
        Trả về None nếu từ không có trong từ điển của mô hình.
        """
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            print(f"Word '{word}' not found in vocabulary (OOV).")
            return None
    
    def get_similarity(self, word1: str, word2: str):
        """
        Tính độ tương đồng cosine giữa hai từ.
        """
        if word1 not in self.model.key_to_index:
            print(f"'{word1}' is OOV (Out Of Vocabulary).")
            return None
        if word2 not in self.model.key_to_index:
            print(f"'{word2}' is OOV (Out Of Vocabulary).")
            return None
        
        similarity = self.model.similarity(word1, word2)
        return similarity
    
    def get_most_similar(self, word: str, top_n: int = 10):
        """
        Tìm top N từ có nghĩa gần nhất với từ được cho.
        """
        if word not in self.model.key_to_index:
            print(f"'{word}' is OOV (Out Of Vocabulary).")
            return []
        
        similar_words = self.model.most_similar(word, topn=top_n)
        return similar_words
    
    def embed_document(self, document: str):
        """
        Tạo vector biểu diễn toàn văn bản bằng cách trung bình các word embeddings.
        """
        tokens = self.tokenizer.tokenize(document)
        vectors = []

        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)
        
        # Nếu không có từ nào hợp lệ (OOV hết)
        if not vectors:
            print("Document has no known words; returning zero vector.")
            return np.zeros(self.model.vector_size)
        
        # Tính trung bình theo từng chiều
        doc_embedding = np.mean(vectors, axis=0)
        return doc_embedding
