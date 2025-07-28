#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "load-manager.h"
#include <stdio.h>

__device__ float get_distance_l1(const float* point, const float* centroid, int DIM)
{
    float dist = 0.0f;
    for (int d = 0; d < DIM; ++d)
        dist += fabsf(point[d] - centroid[d]);
    return dist;
}

__device__ float get_distance_l2(const float* point, const float* centroid, int DIM)
{
    float squaredDistance = 0.0f;
    for (int d = 0; d < DIM; ++d) {
        float vecDiff = point[d] - centroid[d];
        squaredDistance += vecDiff * vecDiff;
    }
    return sqrtf(squaredDistance);
}

__device__ float get_distance_cosine(const float* point, const float* centroid, int DIM)
{
    float dot = 0.0f, norm_point = 0.0f, norm_centroid = 0.0f;
    for (int d = 0; d < DIM; ++d) {
        dot += point[d] * centroid[d];
        norm_point += point[d] * point[d];
        norm_centroid += centroid[d] * centroid[d];
    }
    float denom = sqrtf(norm_point) * sqrtf(norm_centroid) + 0.00000001f;
    float cosine_similarity = dot / denom;

    return 1.0f - cosine_similarity;
}

__global__ void distance_l1_kernel(const float* db_vectors, const float* query, float* dists, int N, int DIM) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    dists[idx] = get_distance_l1(db_vectors + idx * DIM, query, DIM);
}

__global__ void distance_l2_kernel(const float* db_vectors, const float* query, float* dists, int N, int DIM) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    dists[idx] = get_distance_l2(db_vectors + idx * DIM, query, DIM);
}

__global__ void distance_cosine_kernel(const float* db_vectors, const float* query, float* dists, int N, int DIM) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    dists[idx] = get_distance_cosine(db_vectors + idx * DIM, query, DIM);
}

void launch_distance_kernel(const float *d_db_vectors, const float *d_query, float *d_dists,
                           int N, int DIM, NormType normType, int TPB)
{
    int numBlocks = (N + TPB - 1) / TPB;
    if (normType == L1_NORM)
        distance_l1_kernel<<<numBlocks, TPB>>>(d_db_vectors, d_query, d_dists, N, DIM);
    else if (normType == L2_NORM)
        distance_l2_kernel<<<numBlocks, TPB>>>(d_db_vectors, d_query, d_dists, N, DIM);
    else
        distance_cosine_kernel<<<numBlocks, TPB>>>(d_db_vectors, d_query, d_dists, N, DIM);
}

__device__ float get_distance_l2_pair(const float* a, const float* b, int DIM) {
    float dist = 0.0f;
    for (int d = 0; d < DIM; ++d) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

__global__ void pairwise_distance_kernel(const float* db_vectors, float* dists, int N, int DIM) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        size_t idx = static_cast<size_t>(i) * N + j;
        if (idx >= static_cast<size_t>(N) * static_cast<size_t>(N)) {
            printf("Index out of bound: i=%d j=%d idx=%zu\n", i, j, idx);
        }
        dists[idx] = get_distance_l2_pair(db_vectors + i * DIM, db_vectors + j * DIM, DIM);
    }
}

void launch_pairwise_distance_kernel(const float *d_db_vectors, float *d_pairwise,
                                     int N, int DIM, NormType normType, int blockX, int blockY) {
    dim3 block(blockX, blockY);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    pairwise_distance_kernel<<<grid, block>>>(d_db_vectors, d_pairwise, N, DIM);
}

__global__ void pairwise_distance_tile_kernel(const float* db_vectors, const float* db_vectors2, float* tile_out, int N, int DIM, int row_offset, int col_offset, int tile_rows, int tile_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < tile_rows && col < tile_cols) {
        int idx_i = row_offset + row;
        int idx_j = col_offset + col;
        tile_out[row * tile_cols + col] =
            get_distance_l2(db_vectors + idx_i * DIM, db_vectors2 + idx_j * DIM, DIM);
    }
}

void launch_pairwise_distance_tile_kernel(const float* d_db_vectors, const float* d_db_vectors2, float* d_tile, int N, int DIM, int row_offset, int col_offset, int tile_rows, int tile_cols, dim3 block, dim3 grid) {
    pairwise_distance_tile_kernel<<<grid, block>>>(d_db_vectors, d_db_vectors2, d_tile, N, DIM, row_offset, col_offset, tile_rows, tile_cols);
}