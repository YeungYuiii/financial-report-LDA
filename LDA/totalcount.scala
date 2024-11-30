import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

// Create Spark session
val spark = SparkSession.builder().appName("CountVectorizer Grouping by Word Length").getOrCreate()

val dfWithFeatures = spark.read.parquet("hdfs://localhost:9000/proj/data/CountVectorizer")
val loadedCountVectorizerModel = CountVectorizerModel.load("hdfs://localhost:9000/proj/data/CountVectorizerModel")
val vocabulary = loadedCountVectorizerModel.vocabulary

// Convert the 'features' column (sparse vector) to a list of words
val wordIndex = countVectorizerModel.vocabulary // Get the vocabulary from the CountVectorizer model

// Define a UDF to extract words from the 'features' column and calculate their lengths
val extractWordLengths = udf((features: org.apache.spark.ml.linalg.SparseVector) => {
  val indices = features.indices
  val wordLengths = indices.map(index => wordIndex(index).length)
  wordLengths
})

// Apply the UDF to extract word lengths
val dfWithWordLengths = dfWithFeatures.withColumn("word_lengths", extractWordLengths(col("features")))

// Flatten the word lengths into individual rows and group by length
val wordLengthDF = dfWithWordLengths
  .select(explode(col("word_lengths")).alias("word_length")) // Flatten the word lengths into individual rows
  .groupBy("word_length")
  .count() // Count occurrences of each word length

// Show the result grouped by word length
wordLengthDF.show()