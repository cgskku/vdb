# GPU-Accelerated Vector Processing Toolkit

This repository provides a collection of **CUDA-accelerated modules** for large-scale vector processing and clustering.  
It includes both the **K-means Clustering** module, optimized for building Inverted File (IVF) indices, and the **Load Manager**, which efficiently handles massive vector data on GPUs.
These components can be used independently or together as part of a GPU-based vector indexing and search pipeline.

## Repository Structure
```plaintext
src/
├── kmeans-vector-clustering/   # CUDA-based K-means clustering for IVF index build
│   ├── kmeans_openai.cpp
│   ├── kmeans.cu
│   ├── README.md
│   └── ...
│
├── load-manager/               # GPU Load Manager with tile-based memory control
    ├── load-manager.cpp
    ├── load-manager.cu
    ├── README.md
    └── ...
