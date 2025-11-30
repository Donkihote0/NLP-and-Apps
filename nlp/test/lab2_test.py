from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.core.dataset_loaders import load_raw_text_data

# Evaluation
def main():
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    doc_term_matrix = vectorizer.fit_transform(corpus)

    print("\n--- CountVectorizer Test ---")
    print("Vocabulary (word -> index):")
    for word, idx in vectorizer.vocabulary_.items():
        print(f"{word}: {idx}")

    print("\nDocument-Term Matrix:")
    for i, vec in enumerate(doc_term_matrix):
        print(f"Doc {i+1}: {vec}")
        
        
    dataset_path = "D:/Studying/NLP/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)  
    
    sample_text = raw_text[:50]
    
    test_sample = vectorizer.fit_transform(sample_text)
    
    print("\nCountVectorizer Test:")
    print("Vocabulary (word -> index):")
    for word, idx in vectorizer.vocabulary_.items():
        print(f"{word}: {idx}")

    print("\nDocument-Term Matrix:")
    for i, vec in enumerate(test_sample):
        print(f"Doc {i+1}: {vec}")

if __name__ == "__main__":
    main()
