import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession

// Create Spark session
val spark = SparkSession.builder().appName("LDA Topic Modeling with TF-IDF").getOrCreate()

// Read the previously saved dfWithTFIDF DataFrame
val dfWithTFIDF = spark.read.parquet("hdfs://localhost:9000/proj/TFIDF")

// Configure and train the LDA model
val lda = new LDA()
  .setK(12)                // Number of topics
  .setMaxIter(500)         // Maximum number of iterations
  .setFeaturesCol("tfidf") // Column containing the TF-IDF features

val ldaModel = lda.fit(dfWithTFIDF)

// Show the topics discovered by the LDA model
val topics = ldaModel.describeTopics(30) // Show the top 5 terms for each topic
topics.show(false)

// Get the topic distribution for each document
val topicDistributions = ldaModel.transform(dfWithTFIDF)

topicDistributions.write.mode("overwrite").parquet("hdfs://localhost:9000/proj/lda_tfidf")

// Stop Spark session
spark.stop()