import nltk
nltk.download('punkt')

# Tokenize theo câu và từ tiếng Anh
from nltk.tokenize import sent_tokenize, word_tokenize
text = "Hello world! NLP is amazing. Let's learn together."
print("Sentence tokenization:")
print(sent_tokenize(text))
print("\nWord tokenization:")
print(word_tokenize(text))

# Áp dụng với tiếng Việt
from underthesea import word_tokenize
text = "Tôi đang học xử lý ngôn ngữ tự nhiên bằng Python."
print(word_tokenize(text))

# stopword
from nltk.corpus import stopwords
nltk.download('stopwords')
text = "This is an example showing how to remove stopwords."
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
filtered = [w for w in words if w.lower() not in stop_words]
print(filtered)

stopwords_vi = set(["và", "là", "các", "một", "những", "được", "bị"])
text = "Học sinh là những người đang được đào tạo trong nhà trường."
tokens = word_tokenize(text)
filtered = [t for t in tokens if t not in stopwords_vi]
print(filtered)

# stemming và lemmatization
# stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
words = ["studies", "studying", "studied"]
print([ps.stem(w) for w in words])

# lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lm = WordNetLemmatizer()
print(lm.lemmatize("studies", pos="v"))

# chuẩn hóa
import regex as re
import unicodedata

def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"[^0-9a-zA-Záàảãạâầấẩẫậăằắẵẳặéèẻẽẹêềếểễệíìỉĩịóòỏõọôồốổỗộơờớởỡợúùủũụưừứửữựýỳỷỹỵđ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print(normalize("Tôi   đang   HỌC  xử-lý ngôn ngữ !!! tự nhiên…"))

# POS Tagging
from underthesea import pos_tag
text = "Tôi đang học xử lý ngôn ngữ tự nhiên."
print(pos_tag(text))
