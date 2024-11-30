import org.apache.spark.ml.feature.{CountVectorizer, IDF, CountVectorizerModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

// Create Spark session
val spark = SparkSession.builder().appName("Text Cleaning and Vectorization").getOrCreate()
val df = spark.read.parquet("hdfs://localhost:9000/proj/cleaned")

// Convert file_name to id
val fileNameToId = df.select("file_name").distinct()
  .rdd.map(_.getString(0)).zipWithIndex().collectAsMap()

val convertFileNameToId = udf((fileName: String) => fileNameToId(fileName).toInt)
val dfWithId = df.withColumn("id", convertFileNameToId(col("file_name")))

// Apply CountVectorizer on the 'all_words' column to get word counts
val countVectorizer = new CountVectorizer()
  .setInputCol("all_words")  // Input column should be an array of words
  .setOutputCol("features")  // Output column will contain the vector of word counts
  .setVocabSize(3000)       // Limit the size of the vocabulary
  .setMaxDF(1350)
  .setMinDF(10)
  .setMinTF(2)

val countVectorizerModel: CountVectorizerModel = countVectorizer.fit(dfWithId)
countVectorizerModel.write.overwrite().save("hdfs://localhost:9000/proj/CountVectorizerModel")

// Transform the data into word counts (sparse vector representation)
val dfWithFeatures = countVectorizerModel.transform(dfWithId)
dfWithFeatures.write.mode("overwrite").parquet("hdfs://localhost:9000/proj/CountVectorizer")

dfWithFeatures.show(5)

// Now compute the TF-IDF for the word counts
val idf = new IDF()
  .setInputCol("features")   // Input column is the word count vector (features)
  .setOutputCol("tfidf")     // Output column will contain the TF-IDF vectors

// Fit and transform the data to compute the TF-IDF
val dfWithTFIDF = idf.fit(dfWithFeatures).transform(dfWithFeatures)
dfWithTFIDF.write.mode("overwrite").parquet("hdfs://localhost:9000/proj/TFIDF")

// Stop Spark session
spark.stop()
