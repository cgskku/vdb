# CUDA K-Means Clustering with GPU Acceleration for High-Dimensional Data

This project leverages CUDA-based GPU acceleration to optimize vector database operations, focusing on index building performance.

---

## Features

- **Efficient Data Partitioning and Clustering**  
  Implements advanced techniques commonly used in IVF (Inverted File) indexing for faster and more accurate data handling.

- **Out-of-Core Dataset Optimization**  
  Accelerates data loading and processing for large datasets that exceed GPU memory capacity.

- **High-Dimensional Data Support**  
  Optimized for managing and clustering high-dimensional vectors with scalable GPU parallelism.

---

## Implementation Details

The core implementation uses CUDA to parallelize key operations such as:

- Distance calculations between data points and cluster centroids  
- Cluster assignment and centroid updates using GPU threads  
- Reduction and aggregation for efficient mean calculations  

---

## KMeans Data Structure

Original CUDA K-means is an optimized implementation that parallelizes the traditional K-means algorithm using CUDA.

### Main Characteristics

- **Data Parallelism Utilization**
  - Cluster centroids remain fixed during each iteration, ensuring safe concurrent access from multiple threads.  
  - Parallel processing of distance calculations for data points to enhance performance.

- **Computation Optimization**
  - Safe parallel processing through atomic operations in distance calculation and centroid update phases.  
  - Efficient workload distribution using block and thread structure.

- **Input Parameters**
  - Dataset with D-dimensional coordinates  
  - Number of clusters (K)  
  - Threads per block (TPB)  
  - Maximum number of iterations  

This implementation provides significantly improved clustering performance compared to CPU-based approaches.

---

## KMeans_dim Data Structure

In GPU-based parallel K-means, each thread (representing a data point) requires repeated access to centroid data stored in global memory.  
Global memory access overhead increases with data dimensionality, so optimization is required.

### Key Features

- **Shared Memory Optimization**
  - Each thread block loads centroids into shared memory to minimize global memory access latency.  
  - Threads within the same block can efficiently share and reuse centroid data.

- **Dimension Chunk Strategy**
  - High-dimensional data is processed in fixed-size chunks (256 dimensions per chunk).  
  - This chunking approach overcomes shared memory size limitations (48KB).  
  - Partial distance calculations are performed per chunk and accumulated for final results.

---