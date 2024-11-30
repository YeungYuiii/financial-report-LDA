import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

// Create Spark session
val spark = SparkSession.builder().appName("Text Cleaning").getOrCreate()
val hdfsPath = "hdfs://localhost:9000/proj/*.txt"
val textFiles = spark.sparkContext.wholeTextFiles(hdfsPath)

// Load the text files from HDFS into a DataFrame and include the file name as the ID
val df = spark.read.text(hdfsPath)
  .withColumn("file_name", input_file_name())  // Add the file name as a column
  .toDF("text_column", "file_name")

val dfWithFileName = df.withColumn("file_name", regexp_extract(col("file_name"), "([^/]+$)", 0))

// Define a UDF to clean the text (simplified for debugging)
val cleanTextUdf = udf((text: String) => {
  if (text == null) "" // Return empty string if null
  else {
    val cleanedText = text                  // Simple cleaning: Remove Chinese characters and digits
      .replaceAll("[\\p{P}]", "")           // Remove all punctuation
      .replaceAll("[^\\p{L}\\s]", "")
      .replaceAll("[\\u4e00-\\u9fff]", "")  // Remove Chinese characters (Unicode range: 4E00–9FFF, etc.)
      .replaceAll("[0-9]", "")              // Remove numbers
      .trim                                 // Remove leading/trailing spaces

    cleanedText.toLowerCase
  }
})

// Apply the UDF to the DataFrame to clean the text
val dfCleaned = dfWithFileName.withColumn("cleaned_text", cleanTextUdf(col("text_column")))

// Split the cleaned text into an array of words
val dfWithWords = dfCleaned.withColumn("words", split(col("cleaned_text"), "\\s+"))

// Remove stopwords from the cleaned text (which is now an array of words)
val remover = new StopWordsRemover()
  .setInputCol("words")  // Input column should be an array of words
  .setOutputCol("filtered_words")

val dfNoStopWords = remover.transform(dfWithWords)

// Filter out words with length less than 3
val dfFilteredWords = dfNoStopWords.withColumn(
  "filtered_words", 
  expr("filter(words, word -> length(word) >= 3)")
)

// Now calculate the word lengths for the filtered words
val dfWithWordLengths = dfFilteredWords.withColumn(
  "word_lengths", 
  expr("transform(filtered_words, word -> length(word))")
)

// Now group by 'file_name' and aggregate all words into a single row per document
val dfGrouped = dfWithWordLengths
  .groupBy("file_name")
  .agg(
    collect_list("filtered_words").alias("all_words"),       // Collect all filtered words
    collect_list("word_lengths").alias("all_word_lengths")   // Collect all word length arrays
  )  

// Flatten the 'all_words' column (which is a vector of vectors) into a single vector
val dfFlattened = dfGrouped.withColumn("all_words", flatten(col("all_words")))
val dfFlattenedWithLengths = dfFlattened.withColumn("all_word_lengths", flatten(col("all_word_lengths")))

dfFlattenedWithLengths.show(5)

// Save the final dfFlattened DataFrame to HDFS or any other format
val outputPath = "hdfs://localhost:9000/proj/cleaned"
dfFlattenedWithLengths.write.mode("overwrite").parquet(outputPath)
