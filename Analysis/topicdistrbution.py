import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("LDA Topic Modeling").getOrCreate()

# Load the LDA result from the Parquet file
lda_result = spark.read.parquet("hdfs://localhost:9000/proj/lda_tf")

# Sample a subset of the DataFrame (e.g., 10% of the data)
sampled_lda_result = lda_result.sample(fraction=0.8, seed=1234)  # Adjust the fraction as needed

# Convert the sampled DataFrame to Pandas for easier manipulation
lda_result_pandas = sampled_lda_result.select("topicDistribution").toPandas()

# Extract the topic distributions (which are stored as DenseVector)
topic_distributions = lda_result_pandas['topicDistribution'].apply(lambda x: x.toArray())

# Convert the topic distributions into a DataFrame where each column corresponds to a topic
topic_df = pd.DataFrame(topic_distributions.tolist())

# Plot the boxplot for each topic
plt.figure(figsize=(10, 6))

sns.violinplot(data=topic_df, inner="point", scale="area")
# Set the y-axis limits to show only topic proportions from 0 to 0.2
plt.grid(False)
plt.title("Topic Distribution Across Sampled Documents")
plt.xlabel("Topic Index")
plt.ylabel("Topic Proportion")
plt.xticks(rotation=90)
plt.savefig('chart.png', bbox_inches='tight')

# Stop the Spark session
spark.stop()
