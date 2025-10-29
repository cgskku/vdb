#ifndef KMEANS_H
#define KMEANS_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_kmeans_labeling(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension);
void launch_kmeans_update_center(float *d_datapoints, int *d_clust_assn, float *d_partial_centroids, int *d_partial_clust_sizes, int N, int TPB, int K, int dimension);
void launch_kmeans_average_centers(float *d_clusterCenters, int *d_clusterSizes, int K, int dimension, int TPB);

// Tile-based functions
void launch_kmeans_labeling_tile(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension, int tile_start, int tile_size);
void launch_kmeans_update_center_tile(float *d_datapoints, int *d_clust_assn, float *d_partial_centroids, int *d_partial_clust_sizes, int N, int TPB, int K, int dimension, int tile_start, int tile_size);

// Corner turning functions
void transpose_centers(float *d_centroids, float *d_centroids_T, int K, int dimension, int TPB);
void launch_kmeans_labeling_corner_turning(float *d_datapoints, int *d_clust_assn, float *d_centroids_T, int N, int TPB, int K, int dimension);
void launch_kmeans_labeling_corner_turning_tile(float *d_datapoints, int *d_clust_assn, float *d_centroids_T, int N, int TPB, int K, int dimension, int tile_start, int tile_size);

// Stream-based functions
void launch_kmeans_labeling_corner_turning_tile_stream(float *d_datapoints, int *d_clust_assn, float *d_centroids_T, int N, int TPB, int K, int dimension, int tile_start, int tile_size, cudaStream_t stream);
void launch_kmeans_labeling_tile_stream(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension, int tile_start, int tile_size, cudaStream_t stream);
void launch_kmeans_update_center_tile_stream(float *d_datapoints, int *d_clust_assn, float *d_partial_centroids, int *d_partial_clust_sizes, int N, int TPB, int K, int dimension, int tile_start, int tile_size, cudaStream_t stream);
void launch_kmeans_average_centers_stream(float *d_clusterCenters, int *d_clusterSizes, int K, int dimension, int TPB, cudaStream_t stream);
void transpose_centers_stream(float *d_centroids, float *d_centroids_T, int K, int dimension, int TPB, cudaStream_t stream);

// cuBLAS-based functions
void launch_kmeans_labeling_cublas(float* d_datapoints, int* d_assign, float* d_centroids, 
                                  float* d_distances, int N, int K, int DIM, cublasHandle_t handle);
#ifdef __cplusplus
}
#endif

#endif