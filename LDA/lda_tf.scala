import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

// Create Spark session
val spark = SparkSession.builder().appName("LDA Topic Modeling").getOrCreate()

// Read the previously saved dfWithFeatures DataFrame
val dfWithFeatures = spark.read.parquet("hdfs://localhost:9000/proj/CountVectorizer")

// Define the number of topics (K) you want to extract
val numTopics = 12

// Perform LDA for topic modeling
val lda = new LDA()
  .setK(numTopics)                // Set the number of topics
  .setMaxIter(500)                 // Set the number of iterations for the algorithm
  .setFeaturesCol("features")     // Use the 'features' column for word counts
  .setSeed(1234)                  // Optional: set a seed for reproducibility

// Fit the LDA model
val ldaModel = lda.fit(dfWithFeatures)

// Transform the data (apply the LDA model to get topic distributions)
val ldaResult = ldaModel.transform(dfWithFeatures)

// Optionally, display the topics (the words associated with each topic)
val topics = ldaModel.describeTopics(30)  // Show top 10 words for each topic
topics.show(false)  // Set `false` to show full topic words
ldaResult.write.mode("overwrite").parquet("hdfs://localhost:9000/proj/lda_tf")

// Stop Spark session
spark.stop()
