import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

// Create Spark session
val spark = SparkSession.builder().appName("LDA Topic Modeling Extraction").getOrCreate()

// Read the LDA result from the Parquet file
val ldaResult = spark.read.parquet("hdfs://localhost:9000/proj/lda_tf")

// Show the schema to see the available columns
ldaResult.printSchema()

// Extract topic distribution from the 'topicDistribution' column
// Convert the sparse vector to a dense vector or an array for better readability
val topicDistributionDF = ldaResult
  .withColumn("topicDistributionArray", udf((topicDist: DenseVector) => topicDist.toArray).apply(col("topicDistribution")))

// Show the topic distribution for each document (and optionally limit it to the first few rows)
topicDistributionDF.select("id", "topicDistributionArray").show(false)
topicDistributionDF.write.mode("overwrite").parquet("hdfs://localhost:9000/lda_topic_distribution")

// Stop Spark session
spark.stop()
