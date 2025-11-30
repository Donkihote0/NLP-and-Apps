import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split

def main():

    # Khởi tạo Spark Session

    spark = (
        SparkSession.builder
        .appName("Spark Word2Vec Demo")
        .master("local[*]")  # chạy trên tất cả CPU cores của máy hiện tại
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("Spark session started.")

    # Đọc dữ liệu JSON
    data_path = "D:/Studying/NLP/c4-train.00000-of-01024-30K.json.gz" # đường dẫn tuyệt đối, có thể tải file data về và thay đổi đường dẫn

    print(f"\n Loading dataset from {data_path} ...")
    df = spark.read.json(data_path)

    # Kiểm tra cấu trúc dữ liệu
    if "text" not in df.columns:
        print("The dataset does not contain a 'text' field.")
        spark.stop()
        return

    print(f"Loaded {df.count()} documents with 'text' field.")

    # Tiền xử lý văn bản
    print("\nPreprocessing text...")

    # Chỉ lấy cột text, chuyển về chữ thường và loại bỏ ký tự đặc biệt
    df_clean = (
        df.select(lower(col("text")).alias("text"))
        .withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", " "))
        .withColumn("text", regexp_replace(col("text"), "\\s+", " "))
    )

    # Tokenize thành danh sách từ
    df_tokens = df_clean.withColumn("words", split(col("text"), " "))

    # Huấn luyện Word2Vec
    print("\nTraining Word2Vec model...")

    word2vec = Word2Vec(
        vectorSize=100,  # 100 chiều
        minCount=5,      # bỏ từ xuất hiện ít hơn 5 lần
        inputCol="words",
        outputCol="features"
    )

    model = word2vec.fit(df_tokens)

    print("\nModel trained successfully!")

    # SHow mô hình
    target_word = "computer"
    print(f"\n Top 5 words similar to '{target_word}':")

    try:
        synonyms = model.findSynonyms(target_word, 5)
        for row in synonyms.collect():
            print(f"  {row['word']:<10} -> {row['similarity']:.4f}")
    except Exception as e:
        print(f"Word '{target_word}' not found in the vocabulary.")

    # Dừng Spark
    print("\nStopping Spark session...")
    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
