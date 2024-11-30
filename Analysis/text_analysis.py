from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TF Visualization") \
    .getOrCreate()

# Read the aggregated TF data from HDFS
input_path = "hdfs://localhost:9000/proj/dfAggregated"  # Update path if needed
df = spark.read.parquet(input_path)

# Show the data schema to check
df.printSchema()
df.show(5)

# Sort by 'total_tf' in descending order
df_sorted = df.orderBy("document_count", ascending=False)

# Show the sorted data
df_sorted.show(5)

# Convert the sorted DataFrame to Pandas
df_pandas = df_sorted.toPandas()

# Check the first few rows
df_pandas.head()

import random
import matplotlib.pyplot as plt
import seaborn as sns

# Randomly sample 100 rows from the DataFrame (this ensures we have a manageable number for plotting)
df_sample = df_pandas.sample(n=10000, random_state=42)  # 'n' specifies the number of rows to sample

# Create a distribution plot of the 'total_tf' values
plt.figure(figsize=(10, 6))
sns.histplot(df_sample['document_count'], kde=True, color='skyblue', bins=20)

# Set labels and title
plt.xlabel('Total TF')
plt.ylabel('Frequency')
plt.title('Distribution Plot of Total TF (Random Sample)')

# Save the chart to a file (without displaying it interactively)
plt.tight_layout()  # Ensure everything fits without overlap
plt.savefig('linechart.png', bbox_inches='tight')


