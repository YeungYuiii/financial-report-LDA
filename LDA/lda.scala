import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

// Read the Hadoop output (doc.txt:word /t wordcount)
val hadoopOutput: RDD[String] = sc.textFile("hdfs://localhost:9000/proj/tf")

// Parse the Hadoop output
val parsedData = hadoopOutput.map { line =>
  val Array(docWord, count) = line.split("\t")
  val Array(doc, word) = docWord.split(":")
  (doc, word, count.toInt) // (document, word, wordcount)
}

// Group data by document and aggregate word counts
val docWordCounts = parsedData
  .groupBy(_._1) // Group by document name
  .mapValues { wordCounts =>
    // Convert to a Map of word -> count
    wordCounts.map { case (_, word, count) => (word, count) }.toMap
  }

// Create an RDD for LDA input (document ID, term frequency vector)
val ldaInput: RDD[(Long, org.apache.spark.mllib.linalg.Vector)] = docWordCounts.zipWithIndex().map {
  case ((doc, wordCounts), docId) =>
    val wordIndices = wordCounts.keys.zipWithIndex.toMap
    val wordCountsArray = wordCounts.values.toArray
    val sparseVector = Vectors.sparse(wordCounts.size, wordIndices.values.toArray, wordCountsArray.map(_.toDouble))
    (docId, sparseVector)
}

// Now `ldaInput` is in the format (doc_id, term_frequency_vector)

import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vector

val lda = new LDA().setK(10).setMaxIterations(10)
val ldaModel = lda.run(ldaInput)  // Input the term frequency vectors

// You can now use the LDA model to extract topics
val topics = ldaModel.describeTopics(10) // Top 10 terms per topic

// Save the trained LDA model to a directory
ldaModel.save(sc, "hdfs://localhost:9000/proj/lda_model")




