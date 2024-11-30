import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("LDA Topic Modeling").getOrCreate()

# Load the LDA result from the Parquet file
lda_result = spark.read.parquet("hdfs://localhost:9000/proj/lda_tfidf")

# Sample a subset of the DataFrame (e.g., 30% of the data)
sampled_lda_result = lda_result.sample(fraction=0.3, seed=1234)  # Adjust the fraction as needed

# Convert the sampled DataFrame to Pandas for easier manipulation
lda_result_pandas = sampled_lda_result.select("topicDistribution").toPandas()

# Extract the topic distributions (which are stored as DenseVector)
topic_distributions = lda_result_pandas['topicDistribution'].apply(lambda x: x.toArray())

# Convert the topic distributions into a DataFrame where each column corresponds to a topic
topic_df = pd.DataFrame(topic_distributions.tolist())

# Aggregate the mass for each topic (sum of proportions across all sampled documents)
topic_sums = topic_df.sum()

# Plot the bar chart for each topic
plt.figure(figsize=(10, 6))
topic_sums.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Aggregated Topic Distribution Across Sampled Documents")
plt.xlabel("Topic Index")
plt.ylabel("Aggregated Topic Proportion")
plt.xticks(rotation=0)
plt.savefig('chart.png', bbox_inches='tight')


# --- Box Plot for Topic Sparsity ---

# Create a figure for the box plot
plt.figure(figsize=(12, 8))

# Define flier properties to make them more transparent
flier_props = dict(
    marker='o',
    markerfacecolor='blue',
    markersize=5,
    alpha=0.2,  # Set transparency here
    markeredgecolor='blue'
)
# Define flier properties to make them more transparent
flier_props = dict(marker='o',
                   markerfacecolor='blue',
                   markersize=5,
                   alpha=0.2,  # Set transparency here
                   markeredgecolor='blue')

# Create the box plot with transparent fliers
box = topic_df.boxplot(
    figsize=(12, 8),
    vert=True,
    patch_artist=True,
    showfliers=True,  # Ensure fliers are shown
    flierprops=flier_props,  # Apply flier properties
    boxprops=dict(facecolor='lightblue', color='blue'),
    medianprops=dict(color='red'),
    whiskerprops=dict(color='blue'),
    capprops=dict(color='blue')
)

# Optionally, you can remove grid lines for a cleaner look
plt.grid(False)  # Disable grid lines

# Set plot titles and labels
plt.title("Box Plot of Topic Proportions Across Sampled Documents")
plt.xlabel("Topic Index")
plt.ylabel("Topic Proportion")
plt.xticks(rotation=90)
plt.tight_layout()

# Save the box plot
plt.savefig('TopicSparsityBoxPlot_.0-.2.png', bbox_inches='tight')

# Stop the Spark session
spark.stop()
