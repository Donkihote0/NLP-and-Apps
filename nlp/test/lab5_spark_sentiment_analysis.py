from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
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

# Huấn luyện pipeline
model = pipeline.fit(train_data)

# Evaluate the Model
predictions = model.transform(test_data)

# In vài kết quả ví dụ
print("\nSample Predictions")
predictions.select("text", "label", "prediction", "probability").show(5, truncate=80)

# Đánh giá độ chính xác và F1
evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print("\n Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Stop Spark Session
spark.stop()
