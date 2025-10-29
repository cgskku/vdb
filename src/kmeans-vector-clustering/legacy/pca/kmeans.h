#ifndef KMEANS_H
#define KMEANS_H

#ifdef __cplusplus
extern "C" {
#endif
void launch_kmeans_labeling(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension);
void launch_kmeans_update_center(float *d_datapoints, int *d_clust_assn, float *d_partial_centroids, int *d_partial_clust_sizes, int N, int TPB, int K, int dimension);
void launch_kmeans_labeling_chunk(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension, int chunkSize);
void launch_kmeans_update_center_chunk(float *d_datapoints, int *d_clust_assn, float *d_partial_centroids, int *d_partial_clust_sizes, int N, int TPB, int K, int dimension, int chunkSize);

void launch_extract_top_eigenvectors(float* d_Cov, float* d_Vsub, int Dim, int reducedDim, int TPB);
void launch_denormalize(float* d_data, const float* d_mean, const float* d_std, int N, int Dim, int TPB);
#ifdef __cplusplus
}
#endif

#endif