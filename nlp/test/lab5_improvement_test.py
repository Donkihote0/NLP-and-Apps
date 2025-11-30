from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# Load and Prepare Data
data_path = "sentiments.csv"  # File CSV có cột 'text' và 'sentiment'

df = spark.read.csv(data_path, header=True, inferSchema=True)

# Chuẩn hóa nhãn (-1, 1) → (0, 1)
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

# Loại bỏ các hàng null trong cột sentiment
initial_row_count = df.count()
df = df.dropna(subset=["sentiment"])
cleaned_row_count = df.count()

print(f"Loaded {initial_row_count} rows, after cleaning: {cleaned_row_count} rows.")

# Build Preprocessing Pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Train the Model
lr = LogisticRegression(
    maxIter=10,
    regParam=0.001,
    featuresCol="features",
    labelCol="label"
)

# Kết hợp toàn bộ pipeline
pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

# Chia train/test
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Tiền xử lý chung (Tokenizer + StopWordsRemover)
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Định nghĩa các pipeline khác nhau
# TF-IDF + Logistic Regression
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
idf = IDF(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

pipeline_lr = Pipeline(stages=[tokenizer, stopwords_remover, hashingTF, idf, lr])

# TF-IDF + Naive Bayes 
nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
pipeline_nb = Pipeline(stages=[tokenizer, stopwords_remover, hashingTF, idf, nb])

# Word2Vec + Logistic Regression 
word2Vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered_words", outputCol="features")
pipeline_w2v = Pipeline(stages=[tokenizer, stopwords_remover, word2Vec, lr])

# Huấn luyện và đánh giá từng mô hình
def evaluate_model(pipeline, train, test, model_name):
    print(f"\n Training {model_name}")
    model = pipeline.fit(train)
    predictions = model.transform(test)

    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")

    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print(f"{model_name} Results:")
    print(f"  ➤ Accuracy: {accuracy:.4f}")
    print(f"  ➤ F1 Score: {f1:.4f}")

    # In vài ví dụ dự đoán
    print("\nSample Predictions:")
    predictions.select("text", "label", "prediction", "probability").show(5, truncate=80)

    return {"model": model_name, "accuracy": accuracy, "f1": f1}


# Đánh giá cả 3 mô hình
results = []
results.append(evaluate_model(pipeline_lr, train_data, test_data, "TF-IDF + Logistic Regression"))
results.append(evaluate_model(pipeline_nb, train_data, test_data, "TF-IDF + Naive Bayes"))
results.append(evaluate_model(pipeline_w2v, train_data, test_data, "Word2Vec + Logistic Regression"))

# Tổng hợp kết quả
print("\n Summary of Model Performance")
for r in results:
    print(f"{r['model']:35s} | Accuracy: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")

# Stop Spark
spark.stop()
