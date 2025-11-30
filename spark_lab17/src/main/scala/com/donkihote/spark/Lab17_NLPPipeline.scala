package com.donkihote.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import java.nio.file.{Paths, Files}
import org.apache.spark.ml.linalg.Vector
import java.nio.charset.StandardCharsets


object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder()
      .appName("Lab17 NLP Pipeline")
      .master("local[*]") // chạy local
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // Update:
    // Time measure
    def time[R](block: => R, stage: String): R = {
      val t0 = System.nanoTime()
      val result = block
      val t1 = System.nanoTime()
      println(s"Stage [$stage] finished in ${(t1 - t0) / 1e9} sec")
      result
    }

    // 1. Load Data 
    val limit = 2000 // định nghĩa giới hạn 
    val inputPath = "data/c4-train.00000-of-01024-30K.json.gz"
    val df = time({spark.read.json(inputPath).limit(limit)}, "Load Data") // giới hạn limit dòng
    println(s"Loaded ${df.count()} rows")

    // Cột text chính trong C4 dataset
    val textCol = "text"


    // 2. Tokenization 
    val tokenizer = new RegexTokenizer()
      .setInputCol(textCol)
      .setOutputCol("tokens")
      .setPattern("\\W+") // tách theo ký tự không phải chữ bằng biểu thức chính quy

    // (Nếu muốn đổi sang Tokenizer thường, thay bằng:)
    // val tokenizer = new Tokenizer().setInputCol(textCol).setOutputCol("tokens")

    //  3. Stop Word Removal
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // 4. TF-IDF Vectorization and Normalizer 
    val hashingTF = new HashingTF()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setNumFeatures(20000)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("tfidf")

    val normalizer = new Normalizer()
      .setInputCol("tfidf")
      .setOutputCol("normFeatures")
      .setP(2.0) 

    // 5. Pipeline 
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, normalizer))

    val model = time({pipeline.fit(df)}, "Train Pipeline") // Bổ sung thêm bộ đo thời gian
    val result = time({model.transform(df).cache()}, "Transform Data") // bổ sung bộ đo thời gian


    // 6. Save Results
    val outputPath = "results/lab17_pipeline_output.txt"
    val resultsDir = Paths.get("results")
    if (!Files.exists(resultsDir)) Files.createDirectories(resultsDir) // tạo file output nếu file chưa tồn tại

    val textOut = result.select("normFeatures").take(20).map(_.toString()).mkString("\n")
    Files.write(Paths.get(outputPath), textOut.getBytes(StandardCharsets.UTF_8))

    println(s"Pipeline completed. Results saved at $outputPath")

    // 7. Log 
    val logDir = Paths.get("log")
    if (!Files.exists(logDir)) Files.createDirectories(logDir) // tạo file log nếu file chưa tồn tại
    val logPath = logDir.resolve("lab17_log.txt")
    val logMsg = s"""
      |Bắt đầu: ${java.time.Instant.now}
      |Processed rows: ${df.count()}
      |Saved output to: $outputPath
      |Hoàn thành: ${java.time.Instant.now}
      |""".stripMargin
    Files.write(logPath, logMsg.getBytes(StandardCharsets.UTF_8))

    // 8. Tìm kiếm tương đương
    def cosineSim(v1: Vector, v2: Vector): Double = {
      val arr1 = v1.toArray
      val arr2 = v2.toArray
      arr1.zip(arr2).map { case (a, b) => a * b }.sum
    } // vì đã chuẩn hóa nên cos giữa 2 vecto chính là tích vô hướng

    val firstDoc = result.select("text", "normFeatures").head()
    val firstText = firstDoc.getString(0)
    val firstVec = firstDoc.getAs[Vector]("normFeatures")

    val bcFirst = spark.sparkContext.broadcast(firstVec.toArray)

    val sims = result.select("text", "normFeatures").rdd.map { row =>
      val txt = row.getString(0)
      val vec = row.getAs[Vector]("normFeatures").toArray
      val dot = bcFirst.value.zip(vec).map { case (a, b) => a * b }.sum
      (txt, dot)
    }
      .sortBy(-_._2)
      .take(5)

    println(s"\nTop 5 similar docs to: ${firstText.take(200)}...")
    sims.foreach { case (txt, score) =>
      println(f"sim=$score%.4f | ${txt.take(200)}")
    }
    spark.stop()
  }
}


