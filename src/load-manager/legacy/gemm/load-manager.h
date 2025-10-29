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

void launch_pairwise_distance_tile_kernel(const float* d_db_vectors, const float* d_db_vectors2, float* d_tile, int N, int DIM, int row_offset, int col_offset, int tile_rows, int tile_cols, dim3 block, dim3 grid);

void launch_pairwise_distance_tile_kernel_transpose(const float* d_db_vectors, const float* d_db_vectors2_T, float* d_tile, int N, int DIM, int row_offset, int tile_rows, int tile_cols, dim3 block, dim3 grid);

void launch_compute_row_norm2_all(const float* d_src, float* d_out, int N, int D, int TPB);
void launch_transpose_rowmajor_to_colmajor(const float* d_src, float* d_dst, int N, int D);
void launch_fuse_norms_and_dot_to_dist2_colmajor(const float* d_row2, const float* d_col2, float* d_C, int M, int Ncol, int ldc);

#ifdef __cplusplus
}
#endif

#endif