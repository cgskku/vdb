#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "load-manager.h"
#include <stdio.h>

#ifndef TILE_DIM
#define TILE_DIM 32
#endif
#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif

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

void launch_pairwise_distance_kernel(const float *d_db_vectors, float *d_pairwise, int N, int DIM, NormType normType, int blockX, int blockY) {
    dim3 block(blockX, blockY);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    pairwise_distance_kernel<<<grid, block>>>(d_db_vectors, d_pairwise, N, DIM);
}

__global__ void pairwise_distance_tile_kernel(const float* db_vectors, const float* db_vectors2, float* tile_out, int N, int DIM, int row_offset, int col_offset, int tile_rows, int tile_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < tile_rows && col < tile_cols) {
        int idx_i = row_offset + row;
        int idx_j = col_offset + col;
        tile_out[row * tile_cols + col] =
            get_distance_l2(db_vectors + idx_i * DIM, db_vectors2 + idx_j * DIM, DIM);
    }
}

__global__ void pairwise_distance_tile_kernel_transpose(
    const float* db_vectors,      // [N x DIM], row-major
    const float* db_vectors2_T,   // [DIM x tile_cols], col-major
    float* tile_out,
    int N, int DIM,
    int row_offset, int col_offset, int tile_rows, int tile_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < tile_rows && col < tile_cols) {
        int idx_i = row_offset + row;
        int idx_j = col;
        float dist = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            float a = db_vectors[idx_i * DIM + d];          // row vector
            float b = db_vectors2_T[d * tile_cols + idx_j]; // col vector
            float diff = a - b;
            dist += diff * diff;
        }
        tile_out[row * tile_cols + col] = sqrtf(dist);
    }
}

void launch_pairwise_distance_tile_kernel(
    const float* d_db_vectors, const float* d_db_vectors2,
    float* d_tile,
    int N, int DIM,
    int row_offset, int col_offset, int tile_rows, int tile_cols,
    dim3 block, dim3 grid)
{
    pairwise_distance_tile_kernel<<<grid, block>>>(
        d_db_vectors, d_db_vectors2, d_tile, N, DIM,
        row_offset, col_offset, tile_rows, tile_cols);
}

void launch_pairwise_distance_tile_kernel_transpose(
    const float* d_db_vectors, const float* d_db_vectors2_T,
    float* d_tile,
    int N, int DIM,
    int row_offset, int tile_rows, int tile_cols,
    dim3 block, dim3 grid)
{
    pairwise_distance_tile_kernel_transpose<<<grid, block>>>(
        d_db_vectors, d_db_vectors2_T, d_tile, N, DIM,
        row_offset, 0, tile_rows, tile_cols);
}

__global__ void row_norm2_kernel(const float* __restrict__ src, float* __restrict__ out, int N, int D) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    const float* v = src + (size_t)i * D;
    float s = 0.0f;
    for (int d = 0; d < D; ++d) {
        float x = v[d];
        s += x * x;
    }
    out[i] = s;
}
void launch_compute_row_norm2_all(const float* d_src, float* d_out, int N, int D, int TPB) {
    int blocks = (N + TPB - 1) / TPB;
    row_norm2_kernel<<<blocks, TPB>>>(d_src, d_out, N, D);
}

__global__ void transpose_row2col_kernel(const float* __restrict__ src, float* __restrict__ dst, int N, int D) {
    // src: row-major [N x D] => src[i*D + d]
    // dst: col-major [D x N] => dst[d + i*D] (ld = D)
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; // bank conflict 회피

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // d
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // i

    // read tile
    for (int r = 0; r < TILE_DIM; r += BLOCK_ROWS) {
        int yy = y + r;
        if (x < D && yy < N) {
            tile[threadIdx.y + r][threadIdx.x] = src[yy * D + x];
        }
    }
    __syncthreads();

    // write transposed tile
    int tx = blockIdx.y * TILE_DIM + threadIdx.x; // new d
    int ty = blockIdx.x * TILE_DIM + threadIdx.y; // new i
    for (int r = 0; r < TILE_DIM; r += BLOCK_ROWS) {
        int tty = ty + r;
        if (tx < N && tty < D) {
            // dst is [D x N] col-major, ld = D
            dst[tty + tx * D] = tile[threadIdx.x][threadIdx.y + r];
        }
    }
}
void launch_transpose_rowmajor_to_colmajor(const float* d_src, float* d_dst, int N, int D) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((D + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    transpose_row2col_kernel<<<grid, block>>>(d_src, d_dst, N, D);
}

__global__ void fuse_norms_dot2dist2_colmajor_kernel(const float* __restrict__ row2, const float* __restrict__ col2, float* __restrict__ C, int M, int Ncol, int ldc) {
    int r = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    int c = blockIdx.x * blockDim.x + threadIdx.x; // 0..Ncol-1
    if (r < M && c < Ncol) {
        float dot = C[r + c * ldc];
        C[r + c * ldc] = row2[r] + col2[c] - 2.0f * dot;
    }
}
void launch_fuse_norms_and_dot_to_dist2_colmajor(const float* d_row2, const float* d_col2, float* d_C, int M, int Ncol, int ldc) {
    dim3 block(32, 8);
    dim3 grid((Ncol + block.x - 1) / block.x, (M    + block.y - 1) / block.y);
    fuse_norms_dot2dist2_colmajor_kernel<<<grid, block>>>(d_row2, d_col2, d_C, M, Ncol, ldc);
}