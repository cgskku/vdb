#ifndef KMEANS_BANK_H
#define KMEANS_BANK_H

#ifdef __cplusplus
extern "C" 
{
#endif

void launch_kmeans_labeling(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clusterSizes, int N, int TPB, int K, int dimension);
void launch_kmeans_update_center(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clusterSizes, int N, int TPB, int K, int dimension);
void launch_kmeans_average_centers(float *d_clusterCenters, int *d_clusterSizes, int K, int dimension, int TPB);

// chunk functions
void launch_kmeans_labeling_chunk(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension, int chunkSize);
void launch_kmeans_update_center_chunk(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clusterSizes, int N, int TPB, int K, int dimension, int chunkSize);

// tile functions
void launch_kmeans_labeling_tile(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension, int tile_start, int tile_size);
void launch_kmeans_update_center_tile(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clusterSizes, int N, int TPB, int K, int dimension, int tile_start, int tile_size);

// corner turning functions
void transpose_centers(float *d_centroids, float *d_centroids_T, int K, int dimension, int TPB);
void launch_kmeans_labeling_corner_turning_tile(float *d_datapoints, int *d_clust_assn, float *d_centroids_T, int N, int TPB, int K, int dimension, int tile_start, int tile_size);
void transpose_data(float* src, float* dst, int rows, int cols, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#endif