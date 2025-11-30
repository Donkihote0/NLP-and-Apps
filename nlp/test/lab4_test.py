from src.representations.word_embedder import WordEmbedder
import numpy as np

def main():
    print("LAB 4: Word Embedding Exploration")

    # Instantiate the WordEmbedder
    embedder = WordEmbedder("glove-wiki-gigaword-50")

    # Get the vector for 'king'
    king_vec = embedder.get_vector("king")
    print("\nVector for 'king':")
    print(king_vec)
    print("Vector shape:", king_vec.shape)

    # Get similarity between words
    sim_king_queen = embedder.get_similarity("king", "queen")
    sim_king_man = embedder.get_similarity("king", "man")
    print(f"\nSimilarity between 'king' and 'queen': {sim_king_queen:.4f}")
    print(f"Similarity between 'king' and 'man': {sim_king_man:.4f}")

    # Get 10 most similar words to 'computer'
    similar_to_computer = embedder.get_most_similar("computer", top_n=10)
    print("\nTop 10 words similar to 'computer':")
    for word, score in similar_to_computer:
        print(f"  {word:12s} -> {score:.4f}")

    # Embed a sentence
    sentence = "The queen rules the country."
    doc_vector = embedder.embed_document(sentence)
    print("\nDocument embedding for:", sentence)
    print("Vector shape:", doc_vector.shape)
    print("First 10 dimensions:", np.round(doc_vector[:10], 4))

if __name__ == "__main__":
    main()
