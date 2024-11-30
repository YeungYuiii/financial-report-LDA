import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

// Create Spark session
val spark = SparkSession.builder().appName("TF Aggregation READ").getOrCreate()
val df = spark.read.parquet("hdfs://localhost:9000/proj/dfAggregated")

df.show(100)

// Stop Spark session
spark.stop()