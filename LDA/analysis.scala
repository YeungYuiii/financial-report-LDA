import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession


// Create Spark session
val spark = SparkSession.builder().appName("CountVectorizer Word Index").getOrCreate()

// Load the previously saved CountVectorizerModel
val loadedCountVectorizerModel = CountVectorizerModel.load("hdfs://localhost:9000/proj/CountVectorizerModel")

// Read the previously saved dfWithFeatures DataFrame
val dfWithFeatures = spark.read.parquet("hdfs://localhost:9000/proj/CountVectorizer")

// Access the vocabulary from the CountVectorizerModel
val vocabulary = loadedCountVectorizerModel.vocabulary

// Print the vocabulary
// println("Vocabulary: " + vocabulary.mkString(", "))

// Example: Retrieve the word for a given index
val array2D = Array(
      Array(114, 254, 826, 983, 1108, 1028, 4, 1662, 1826, 898, 3, 19, 38, 2551, 1635, 1593, 2174, 106, 2004, 2393),
      Array(89, 85, 197, 425, 386, 483, 170, 919, 315, 29, 1061, 1201, 1989, 1477, 1138, 1378, 375, 1880, 2319, 1452),
      Array(71, 33, 212, 593, 656, 248, 1519, 1901, 544, 188, 1204, 1983, 837, 2221, 757, 1250, 11, 2453, 2137, 2703),
      Array(299, 286, 113, 727, 1170, 84, 1507, 232, 1649, 1987, 1100, 1744, 893, 1773, 2515, 57, 281, 1406, 662, 676),
      Array(9, 94, 552, 24, 68, 830, 998, 1096, 1093, 466, 1334, 363, 1203, 21, 39, 1741, 1352, 1572, 27, 2029),
      Array(160, 319, 253, 36, 1333, 2237, 2236, 1810, 2215, 214, 549, 1850, 773, 40, 6, 1685, 30, 1219, 1916, 616),
      Array(46, 97, 321, 52, 725, 10, 643, 17, 120, 166, 66, 840, 749, 960, 171, 464, 217, 2329, 252, 1164),
      Array(329, 181, 1227, 584, 629, 1564, 769, 1887, 27, 1619, 1103, 59, 2795, 1358, 2, 2037, 581, 2868, 289, 930),
      Array(126, 399, 1235, 201, 350, 452, 635, 1213, 124, 1836, 2723, 1703, 1959, 2497, 1476, 49, 1034, 2, 1165, 497),
      Array(233, 205, 269, 476, 108, 270, 99, 479, 314, 173, 825, 1301, 1503, 24, 1554, 1607, 827, 1608, 2660, 168),
      Array(0, 64, 5, 13, 4, 14, 217, 3, 22, 69, 1, 25, 26, 57, 81, 35, 115, 28, 261, 198),
      Array(11, 7, 470, 1327, 1689, 1460, 933, 422, 51, 1543, 1181, 83, 378, 2042, 932, 2075, 1162, 419, 516, 1653)
)

for (i <- 0 until array2D.length) {
  for (j <- 0 until array2D(i).length) {
    println(s"Element at ($i, $j): ${vocabulary(array2D(i)(j))}")
  }
}

// Stop Spark session
spark.stop()