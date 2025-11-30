import os

def load_raw_text_data(filepath: str) -> str:
    abspath = os.path.abspath(filepath)
    if not os.path.exists(abspath):
        raise FileNotFoundError(
            f"Dataset file not found: {abspath}\n"
            "Please check the path or put the file there."
        )
    with open(abspath, "r", encoding="utf-8") as f:
        return f.read()
