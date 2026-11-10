# Vector Database Acceleration Project

This repository contains research and implementation details for GPU-accelerated vector database modules.  
It focuses on high-dimensional vector clustering, distance computation, candidate-list sorting, and asynchronous GPU processing.

---

## 📘 Project Sections

- [CUDA K-Means Clustering](./kmeans.md)  
  GPU-parallelized K-Means algorithm optimized for high-dimensional datasets.

- [Asynchronous Vector Distance & Memory Load Manager](./load_manager.md)  
  Asynchronous CUDA pipeline with dual-stream architecture for large-scale vector distance computation.

- [GPU Top-k Sorting & Asynchronous Task Scheduling](./gpu_sort.md)\
  Scalable candidate-list sorting with workload-aware scheduling, request batching, and reusable transfer-compute pipelines.

---

## 📅 Funding Information

**Project Title:** 초거대 AI 모델의 장기 기억 저장을 위한 벡터 DB 개발  
**Funding Agency:** IITP (SW컴퓨팅산업원천기술개발사업)  
**Grant No.:** RS-2024-00454666  
**Duration:** 2024.7 — 2027.12

---
