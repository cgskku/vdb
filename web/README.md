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

# Asynchronous GPU-Accelerated Vector Distance Computation and Dynamic Memory Load Manager

This project extends the previous year's GPU-based vector clustering work by developing an asynchronous GPU acceleration framework for large-scale vector distance computation.  
The Memory Load Manager maximizes GPU utilization and eliminates CPU–GPU transfer bottlenecks through overlapping computation and data movement.

---

## Features

- **Asynchronous Memory–Compute Pipeline**  
  Dual-stream CUDA architecture allows concurrent data transfer (H2D/D2H) and computation.  
  Streams are synchronized via CUDA events to ensure consistency without blocking.

- **Tile-Based Distance Computation**  
  Uses cuBLAS GEMM operations to compute pairwise vector distances efficiently with limited GPU memory.

- **GPU-Side Matrix Transposition**  
  Adds a GPU kernel to convert row-major to column-major matrices for coalesced memory access.

---

## Implementation Details

- **Memory Transfer Optimization**  
  Asynchronous data movement using pinned memory enables direct DMA transfers, synchronized with CUDA events.

- **GPU Compute Kernel Pipeline**  
  Includes on-device transposition, cuBLAS GEMM for matrix multiplication, and a fused kernel to compute final distances.

---

## Architectural Summary

The asynchronous GPU-accelerated distance computation framework consists of several key components:

- **Memory Load Manager**
  - Controls asynchronous CPU–GPU data transfer using dual CUDA streams (I/O and Compute).  
  - Ensures continuous GPU workload by overlapping data loading and computation.

- **Pinned Memory Buffers**
  - Allocated using `cudaMallocHost` for high-throughput DMA transfers.  
  - Eliminates page-fault latency and improves bandwidth.

- **Dual CUDA Streams**
  - Separate I/O and Compute streams enable simultaneous data transfer and kernel execution.  
  - Stream-level concurrency maximizes utilization.

- **Event-Based Synchronization**
  - Coordinates execution using `cudaEventRecord` and `cudaStreamWaitEvent`.  
  - Enables precise timing control without stalling the pipeline.

- **cuBLAS GEMM Integration**
  - Uses `cublasSgemm()` for tile-wise distance computation.  
  - Achieves efficient large-matrix multiplication with optimized memory access.

- **Tile Partitioning**
  - Divides datasets into smaller submatrices (tiles) for out-of-core computation.  
  - Allows scalability even when data exceeds GPU memory capacity.

- **GPU Transpose Kernel**
  - Converts data from row-major to column-major before GEMM.  
  - Ensures coalesced access for maximum throughput.

---

This modular architecture achieves high concurrency between data movement and computation, resulting in a continuous, high-throughput GPU pipeline suitable for large-scale vector similarity search and clustering.
