#ifndef KMEANS_MINI_H
#define KMEANS_MINI_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_kmeans_labeling(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int N, int S, int TPB, int K, int DIM);
void launch_kmeans_update_center(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int S, int TPB, int K, int DIM);

#ifdef __cplusplus
}
#endif

#endif
