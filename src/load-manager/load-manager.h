#ifndef LOAD_MANAGER
#define LOAD_MANAGER

typedef enum {
    L1_NORM,
    L2_NORM,
    COSINE_DIST
} NormType;

#ifdef __cplusplus
extern "C" {
#endif

void launch_distance_kernel(const float *d_db_vectors, const float *d_query, float *d_dists, int N, int DIM, NormType normType, int TPB);

void launch_pairwise_distance_kernel(const float *d_db_vectors, float *d_pairwise, int N, int DIM, NormType normType, int blockX, int blockY);

#ifdef __cplusplus
}
#endif

#endif