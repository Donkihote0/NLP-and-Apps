from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

# Task 1
tokenizer = SimpleTokenizer()
example = "Hello, world! This is a test."
print("Testing SimpleTokenizer:")
print(tokenizer.tokenize(example))

# Evaluation
def main():
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    print("SimpleTokenizer:")
    simple_tok = SimpleTokenizer()
    for s in sentences:
        print(f"In: {s}")
        print("Out:", simple_tok.tokenize(s))
        print()

    print("RegexTokenizer: ")
    regex_tok = RegexTokenizer()
    for s in sentences:
        print(f"In: {s}")
        print("Out:", regex_tok.tokenize(s))
        print()
        
    # Task 3
    dataset_path = "D:/Studying/NLP/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)


    sample_text = raw_text[:500]

    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")

    simple_tokens = SimpleTokenizer.tokenize(0 ,sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

    regex_tokens = RegexTokenizer.tokenize(0, sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")    


if __name__ == "__main__":
    main()