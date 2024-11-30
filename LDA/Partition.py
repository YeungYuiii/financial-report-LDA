from pyspark.sql import SparkSession
from pyspark.ml.linalg import SparseVector, DenseVector
import numpy as np
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ReadParquetAndPartitionMatrix") \
    .getOrCreate()

# Read the CountVectorizer transformation result from HDFS as a Parquet file
parquet_path = "hdfs://localhost:9000/proj/data/CountVectorizer"
df = spark.read.parquet(parquet_path)

# Ensure the DataFrame has 'id' and 'features' columns
df.show(5)
df.printSchema()

# Collect the transformed data as a list of (id, features) tuples
vectorized_matrix = df.select("id", "features").rdd.map(lambda row: (row["id"], row["features"])).collect()

# Create a list to store non-zero values as (row_id, col_id, value)
sparse_entries = []

# Extract non-zero values from the vectors
for row_id, vector in vectorized_matrix:
    if isinstance(vector, SparseVector):
        for index, value in zip(vector.indices, vector.values):
            sparse_entries.append((int(row_id), int(index), int(value)))
    elif isinstance(vector, DenseVector):
        for index, value in enumerate(vector.values):
            if value != 0.0:  # Only consider non-zero values
                sparse_entries.append((int(row_id), int(index), int(value)))

# Block partitioning parameters
block_size = 3  # Size of each block

# Create output directory if not exists
if not os.path.exists("matrix_blocks_sparse"):
    os.makedirs("matrix_blocks_sparse")

# Partition the sparse matrix into blocks and save
from collections import defaultdict

# Group entries by block
block_dict = defaultdict(list)
for row, col, value in sparse_entries:
    row_block = row // block_size
    col_block = col // block_size
    block_key = (row_block, col_block)
    block_dict[block_key].append((row % block_size, col % block_size, value))

# Write each block to a file
for (row_block, col_block), entries in block_dict.items():
    file_name = f"matrix_blocks_sparse/v_block_{row_block}_{col_block}.txt"
    with open(file_name, "w") as f:
        content = "\n".join([f"{row} {col} {value}" for row, col, value in entries])
        f.write(content)

# Stop the Spark session
spark.stop()
