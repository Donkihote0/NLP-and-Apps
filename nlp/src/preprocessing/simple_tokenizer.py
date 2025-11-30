from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()

        tokens: List[str] = []
        current_token = ""

        for ch in text:
            if ch.isalnum():  
                current_token += ch
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""

                if ch.isspace():
                    continue

                if ch in [".", ",", "!", "?","@","#","$","%","^","&","*"]:
                    tokens.append(ch)

        if current_token:
            tokens.append(current_token)

        return tokens
