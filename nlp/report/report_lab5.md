# Lab 5 Text Classification

### Nguyễn Đức Đạt - 23000109

## Nội dung chính

### Task 1: Data Preparation (with scikit-learn)

- Dataset: use a small, in-memory dataset for simplicity.
- Vectorize: Use TfidfVectorizer (or CountVectorizer) from Lab 3 (or Lab 2)
  to transform these texts into numerical features.

---

### Task 2: TextClassifier Implementation

1. Create the file: src/models/text_classifier.py.
2. Implement the TextClassifier class:
   • The constructor **init**(self, vectorizer: Vectorizer) should accept a
   Vectorizer instance.
   • It should have an attribute \_model to store the trained LogisticRegression
   model from scikit-learn.
3. Implement fit(self, texts: List[str], labels: List[int]):
   • This method will train the classifier.
   • First, use the vectorizer to fit_transform the input texts into a feature matrix
   X.
   • Initialize a LogisticRegression model (e.g., solver='liblinear' for small
   datasets).
   • Train the model using model.fit(X, labels).
4. Implement predict(self, texts: List[str]) -> List[int]:
   • This method will make predictions on new texts.
   • First, use the vectorizer to transform the input texts into a feature matrix X.
   • Use the trained \_model to predict labels: \_model.predict(X).
5. Implement evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str,
   float]:
   • This method will calculate evaluation metrics.
   • Use sklearn.metrics functions (accuracy_score, precision_score, recall_score,
   f1_score) to compute and return a dictionary of metrics.

---

### Task 3: Evaluation

• Create a new test file: test/lab5_test.py.
• Define the texts and labels dataset.
• Split the data into training and testing sets (e.g., 80% train, 20% test). You can
use sklearn.model_selection.train_test_split.
• Instantiate your RegexTokenizer and TfidfVectorizer.
• Instantiate your TextClassifier with the vectorizer.
• Train the classifier using the training data.
• Make predictions on the test data.
• Evaluate the predictions and print the metrics.

**Output:**

```
Evaluation Results
Accuracy: 0.5000
Precision: 0.2500
Recall: 0.5000
F1: 0.3333
```

---

### Advanced Example: Sentiment Analysis with PySpark

1. Initialize Spark Session
2. Load Data: Data is read from the data/sentiments.csv file. This file contains “text”
   and “sentiment” columns.
3. Build Preprocessing Pipeline: A Pipeline in Spark ML consists of a sequence
   of Transformer and Estimator:
   • Tokenizer: Splits text into words (tokens).
   tokenizer = Tokenizer(inputCol="text", outputCol="words")
   • StopWordsRemover: Removes common stop words from the token list.
   stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
   • HashingTF: Converts a set of tokens into a fixed-size feature vector using a
   hashing technique. This is an efficient way to vectorize text.
   hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
   • IDF (Inverse Document Frequency): Rescales the feature vectors
   produced by HashingTF. It down-weights terms that appear frequently in the
   corpus.
   idf = IDF(inputCol="raw_features", outputCol="features")
4. Train the Model:
   • LogisticRegression: The model used for classification.
   lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
   • Assemble the Pipeline: All steps are combined into a single Pipeline.
   pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
   • Training: Call pipeline.fit() on the training data. Spark will automatically
   execute all stages in the pipeline.
   model = pipeline.fit(trainingData)
5. Evaluate the Model:
   • Use model.transform() on the test data to get predictions.
   • MulticlassClassificationEvaluator is used to calculate metrics like accuracy
   and f1.
   evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
   accuracy = evaluator.evaluate(predictions)

**Output:**

```
WARNING: Using incubator modules: jdk.incubator.vector
25/10/28 19:10:10 WARN Shell: Did not find winutils.exe: java.io.FileNotFoundException: java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset. -see https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/10/28 19:10:11 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Loaded 5792 rows, after cleaning: 5791 rows.
25/10/28 19:10:22 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS

Sample Predictions
+--------------------------------------------------------------------------------+-----+----------+------------------------------------------+
|                                                                            text|label|prediction|                               probability|
+--------------------------------------------------------------------------------+-----+----------+------------------------------------------+
|  ISG An update to our Feb 20th video review..if it closes below 495 much low...|  0.0|       1.0|   [0.2614599786780246,0.7385400213219754]|
|  The rodeo clown sent BK screaming into the SI weekly red zone...time to pee...|  0.0|       0.0| [0.9999997991875168,2.008124831975877E-7]|
|                            , ES,SPY, Ground Hog Week, distribution at highs..  |  0.0|       1.0|   [0.0541093238048886,0.9458906761951114]|
|                                                        ES, S  PAT TWO, update  |  0.0|       0.0|[0.9986626113379365,0.0013373886620634545]|
|               PCN doulble top at key fib retracement weekly....time to exit ...|  0.0|       1.0|   [0.4465277905840951,0.5534722094159049]|
+--------------------------------------------------------------------------------+-----+----------+------------------------------------------+
only showing top 5 rows

 Evaluation Results
Accuracy: 0.7295
F1 Score: 0.7266
SUCCESS: The process with PID 11452 (child process of PID 20900) has been terminated.
SUCCESS: The process with PID 20900 (child process of PID 9312) has been terminated.
SUCCESS: The process with PID 9312 (child process of PID 20712) has been terminated.
```

### Task 4: Evaluating and Improving Model Performance

1. Improve Preprocessing and Feature Selection
   The quality of the input features directly impacts model performance.
   • Noise Filtering: Remove special characters, URLs, HTML tags, or other
   meaningless words.
   • Vocabulary Reduction: Limit the vocabulary to include only words that appear
   with a certain frequency (e.g., remove words that are too rare or too common).
   This helps reduce the dimensionality of the feature space and combat noise.
   • Reduce TF-IDF Dimensionality: When using HashingTF in Spark, you can
   experiment with different numFeatures values. A smaller number of features
   might help the model generalize better if the data is noisy.
2. Use Advanced Embedding Methods
   Instead of TF-IDF, we can use dense embeddings to represent text. These embeddings
   often capture the semantics of words better.
   • Word2Vec: Train a Word2Vec model on your text corpus to generate word
   vectors. You can then average the word vectors in a sentence to create a feature
   vector for the sentence. Spark ML provides a Word2Vec implementation.
   • Pre-trained Embeddings: Use pre-trained embeddings like GloVe or FastText.
3. Experiment with More Complex Model Architectures
   The LogisticRegression model is a good baseline, but more complex models can
   capture non-linear relationships in the data.
   • Naive Bayes: A simple probabilistic model that often works well for text tasks.
   • Gradient-Boosted Trees (GBTs): A powerful ensemble model that can yield
   high performance.
   • Neural Networks: These models can learn hierarchical representations of text
   and often give the best results on large datasets.

**Output:**

```
WARNING: Using incubator modules: jdk.incubator.vector
25/10/28 19:11:53 WARN Shell: Did not find winutils.exe: java.io.FileNotFoundException: java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset. -see https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/10/28 19:11:54 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Loaded 5792 rows, after cleaning: 5791 rows.

 Training TF-IDF + Logistic Regression
25/10/28 19:12:04 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
TF-IDF + Logistic Regression Results:
  ➤ Accuracy: 0.7322
  ➤ F1 Score: 0.7306

Sample Predictions:
+--------------------------------------------------------------------------------+-----+----------+-----------------------------------------+
|                                                                            text|label|prediction|                              probability|
+--------------------------------------------------------------------------------+-----+----------+-----------------------------------------+
|  ISG An update to our Feb 20th video review..if it closes below 495 much low...|  0.0|       1.0| [0.26057882224445433,0.7394211777555457]|
|  The rodeo clown sent BK screaming into the SI weekly red zone...time to pee...|  0.0|       0.0| [0.8845374253511644,0.11546257464883558]|
|                            , ES,SPY, Ground Hog Week, distribution at highs..  |  0.0|       1.0| [0.10752192026567267,0.8924780797343274]|
|                                                        ES, S  PAT TWO, update  |  0.0|       0.0|[0.9939625792272556,0.006037420772744384]|
|               PCN doulble top at key fib retracement weekly....time to exit ...|  0.0|       0.0|  [0.7766953540979038,0.2233046459020962]|
+--------------------------------------------------------------------------------+-----+----------+-----------------------------------------+
only showing top 5 rows

 Training TF-IDF + Naive Bayes
TF-IDF + Naive Bayes Results:
  ➤ Accuracy: 0.6907
  ➤ F1 Score: 0.6906

Sample Predictions:
+--------------------------------------------------------------------------------+-----+----------+-------------------------------------------+
|                                                                            text|label|prediction|                                probability|
+--------------------------------------------------------------------------------+-----+----------+-------------------------------------------+
|  ISG An update to our Feb 20th video review..if it closes below 495 much low...|  0.0|       1.0|  [0.008574578596954392,0.9914254214030457]|
|  The rodeo clown sent BK screaming into the SI weekly red zone...time to pee...|  0.0|       0.0|                [1.0,7.464567483085045E-19]|
|                            , ES,SPY, Ground Hog Week, distribution at highs..  |  0.0|       1.0| [5.039441813136982E-14,0.9999999999999496]|
|                                                        ES, S  PAT TWO, update  |  0.0|       0.0|[0.9999999999988904,1.1096627728099956E-12]|
|               PCN doulble top at key fib retracement weekly....time to exit ...|  0.0|       0.0| [0.9974856855262186,0.0025143144737814187]|
+--------------------------------------------------------------------------------+-----+----------+-------------------------------------------+
only showing top 5 rows

 Training Word2Vec + Logistic Regression
Word2Vec + Logistic Regression Results:
  ➤ Accuracy: 0.6664
  ➤ F1 Score: 0.6083

Sample Predictions:
+--------------------------------------------------------------------------------+-----+----------+----------------------------------------+
|                                                                            text|label|prediction|                             probability|
+--------------------------------------------------------------------------------+-----+----------+----------------------------------------+
|  ISG An update to our Feb 20th video review..if it closes below 495 much low...|  0.0|       1.0| [0.28221399498133803,0.717786005018662]|
|  The rodeo clown sent BK screaming into the SI weekly red zone...time to pee...|  0.0|       1.0|[0.29339634087666655,0.7066036591233334]|
|                            , ES,SPY, Ground Hog Week, distribution at highs..  |  0.0|       1.0| [0.3404260183561154,0.6595739816438846]|
|                                                        ES, S  PAT TWO, update  |  0.0|       1.0|   [0.198226387350935,0.801773612649065]|
|               PCN doulble top at key fib retracement weekly....time to exit ...|  0.0|       1.0| [0.3543009315275892,0.6456990684724109]|
+--------------------------------------------------------------------------------+-----+----------+----------------------------------------+
only showing top 5 rows

 Summary of Model Performance
TF-IDF + Logistic Regression        | Accuracy: 0.7322 | F1: 0.7306
TF-IDF + Naive Bayes                | Accuracy: 0.6907 | F1: 0.6906
Word2Vec + Logistic Regression      | Accuracy: 0.6664 | F1: 0.6083

SUCCESS: The process with PID 6172 (child process of PID 22844) has been terminated.
SUCCESS: The process with PID 22844 (child process of PID 26112) has been terminated.
SUCCESS: The process with PID 26112 (child process of PID 13708) has been terminated.
```

---

---

## Nội dung chi tiết

### 1. Các bước thực hiện

1. Xây dựng pineline phân loại văn bản, gồm **Tokenizer**, **Vectorizer**, **Classifier**.
2. Dùng `scikit-learn` cho Logistic Regression model.
3. Dùng `PySpark` với tập train đã đc phân phối trên data.
4. Dùng và cải tiến **Naive Bayes** và tiền xử lí.

---

### 2. Cách chạy

**To run the tests:**

```bash
cd nlp
# Run the basic test
python test/lab5_test.py

# Run the Spark example
python test/lab5_spark_sentiment_analysis.py

# Run the improved model
python test/lab5_improvement_test.py
```

Tất cả đều có các độ đo (accuracy, precision, recall, F1).

---

### 3. Phân tích kết quả

```
TF-IDF + Logistic Regression        | Accuracy: 0.7322 | F1: 0.7306
TF-IDF + Naive Bayes                | Accuracy: 0.6907 | F1: 0.6906
Word2Vec + Logistic Regression      | Accuracy: 0.6664 | F1: 0.6083
```

**Analysis:**

- Ở task 1, 2, 3: data test là do ta tự tạo, rất nhỏ, nên mô có các chỉ số rất thấp, vì ko đủ data huấn luyện
- Với advance task: kết quả thu được khá khả quan, các chỉ số đã có sự cải thiện khi dùng mô hình có sẵn
- Với task 4: Sự cải thiện so với advance task là chưa đáng kể:
  - Mô hình Hồi quy Logistic cơ bản đạt độ chính xác khá ổn (~73%).
  - Phiên bản PySpark cho kết quả tương tự nhưng tốt hơn một chút nhờ tính song song và xử lý TF-IDF hiệu quả.
  - Mô hình Naive Bayes chưa tốt so với Hồi quy Logistic trên tập dữ liệu này. Nó xử lý chưa tốt các đặc trưng thưa thớt và phân phối từ, khiến nó trở nên chưa lý tưởng cho các tác vụ văn bản.
  - Tiền xử lý bổ sung (viết thường, loại bỏ dấu câu) giúp giảm nhiễu và cải thiện khả năng khái quát hóa.

---

### Khó khăn và giải pháp

| Challenge                                   | Solution                                                     |
| ------------------------------------------- | ------------------------------------------------------------ |
| Xử lý dữ liệu văn bản thưa thớt             | Sử dụng vector hóa TF-IDF để biểu diễn văn bản hiệu quả      |
| Overfitting                                 | dùng mô hình đơn giản hơn (Logistic Regression, Naive Bayes) |
| task 2 sai kết quả do tập data test quá nhỏ | thử với dataset lớn hơn                                      |
| Class imbalance                             | áp dụng cân bằng mẫu                                         |

---

### References

- Scikit-learn documentation: https://scikit-learn.org/stable/
- PySpark MLlib documentation: https://spark.apache.org/docs/latest/ml-guide.html
- Text classification tutorial: https://towardsdatascience.com/text-classification-in-python-dd95d264c802
- ChatGPT

---
