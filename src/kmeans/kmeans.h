#ifndef KMEANS_H
#define KMEANS_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_kmeans_labeling(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int TPB, int K, int dimension, int chunkSize);
void launch_kmeans_update_center(float *d_datapoints, int *d_clust_assn, float *d_partial_centroids, int *d_partial_clust_sizes, int N, int TPB, int K, int dimension, int chunkSize);

#ifdef __cplusplus
}
#endif

#endif