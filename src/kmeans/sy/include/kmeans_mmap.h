#ifndef KMEANS_MMAP_H
#define KMEANS_MMAP_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_kmeans_labeling(
    float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clusterSizes,
    int N, int S, int TPB, int K, int dimension, cudaStream_t stream);

void launch_kmeans_update_center(
    float *d_datapoints, int *d_clust_assn, float *d_partial_centroids, int *d_partial_clust_sizes,
    int N, int S, int TPB, int K, int dimension, cudaStream_t stream);

// (선택) 안 쓰면 구현 안 해도 됨
void launch_kmeans_labeling_chunk(float *d_datapoints, int *d_clust_assn, float *d_centroids,
                                  int N, int TPB, int K, int dimension, int chunkSize);
void launch_kmeans_update_center_chunk(float *d_datapoints, int *d_clust_assn,
                                       float *d_partial_centroids, int *d_partial_clust_sizes,
                                       int N, int TPB, int K, int dimension, int chunkSize);

#ifdef __cplusplus
}
#endif

#endif
