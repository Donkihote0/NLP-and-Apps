// Scala version phải khớp với Spark 3.5.1 (hỗ trợ Scala 2.12)
ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "spark-nlp-labs",
    version := "0.1.0",

    //Spark dependencies
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core"  % "3.5.1",
      "org.apache.spark" %% "spark-sql"   % "3.5.1",
      "org.apache.spark" %% "spark-mllib" % "3.5.1"
    ),

    //Sửa lỗi IllegalAccessError khi chạy với Java 17+
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq(
      "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    )
  )








