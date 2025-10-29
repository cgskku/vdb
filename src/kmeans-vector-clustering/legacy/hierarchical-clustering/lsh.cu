#include "lsh.h"

__global__ void lsh_coarse_cluster_kernel(float* d_data, int* d_coarse_labels, const float* d_random_proj, int N, int DIM, int num_buckets) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    float dot = 0.0f;
    for (int d = 0; d < DIM; ++d) dot += d_data[idx * DIM + d] * d_random_proj[d];
    float normalized = fminf(fmaxf((dot + 10.0f) / 20.0f, 0.0f), 1.0f);
    int bucket = (int)(normalized * num_buckets) % num_buckets;
    d_coarse_labels[idx] = bucket;
}

void launch_lsh(float* d_data, int* d_coarse_labels, const float* d_random_proj, int N, int DIM, int num_buckets) {
    dim3 block(256), grid((N + block.x - 1) / block.x);
    lsh_coarse_cluster_kernel<<<grid, block>>>(d_data, d_coarse_labels, d_random_proj, N, DIM, num_buckets);
    cudaDeviceSynchronize();
}

__device__ float clamp(float x, float lower, float upper) {
    return fminf(fmaxf(x, lower), upper);
}

__global__ void multi_bit_lsh_kernel(const float* __restrict__ d_data, int* d_coarse_labels, const float* __restrict__ d_random_proj, int N, int DIM, int NUM_BUCKETS, int N_PROJ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    int hash_code = 0;
    for (int p = 0; p < N_PROJ; ++p) {
        float dot = 0.0f;
        for (int d = 0; d < DIM; ++d)
            dot += d_data[idx * DIM + d] * d_random_proj[p * DIM + d];
        hash_code = (hash_code << 1) | (dot > 0.0f ? 1 : 0);
    }
    int bucket = hash_code % NUM_BUCKETS;
    d_coarse_labels[idx] = bucket;
}

void launch_multi_bit_lsh(const float* d_data, int* d_coarse_labels, const float* d_random_proj, int N, int DIM, int NUM_BUCKETS, int N_PROJ) {
    dim3 block(256), grid((N + block.x - 1) / block.x);
    multi_bit_lsh_kernel<<<grid, block>>>(d_data, d_coarse_labels, d_random_proj, N, DIM, NUM_BUCKETS, N_PROJ);
    cudaDeviceSynchronize();
}

__device__ int flip_bit(int hash_code, int bit_index) {
    return hash_code ^ (1 << bit_index);
}

__global__ void multi_probe_lsh_kernel(const float* __restrict__ d_data, int* d_coarse_labels, const float* __restrict__ d_random_proj, int N, int DIM, int NUM_BUCKETS, int N_PROJ, int N_PROBES) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    int hash_code = 0;
    for (int p = 0; p < N_PROJ; ++p) {
        float dot = 0.0f;
        for (int d = 0; d < DIM; ++d)
            dot += d_data[idx * DIM + d] * d_random_proj[p * DIM + d];
        hash_code = (hash_code << 1) | (dot > 0.0f ? 1 : 0);
    }
    int best_bucket = hash_code % NUM_BUCKETS;
    for (int probe = 0; probe <= N_PROBES && probe <= N_PROJ; ++probe) {
        int probed_code = (probe == 0) ? hash_code : flip_bit(hash_code, probe - 1);
        int candidate_bucket = probed_code % NUM_BUCKETS;
        if (probe == 0 || candidate_bucket < best_bucket) best_bucket = candidate_bucket;
    }
    d_coarse_labels[idx] = best_bucket;
}

void launch_multi_probe_lsh(const float* d_data, int* d_coarse_labels, const float* d_random_proj, int N, int DIM, int NUM_BUCKETS, int N_PROJ, int N_PROBES) {
    dim3 block(256), grid((N + block.x - 1) / block.x);
    multi_probe_lsh_kernel<<<grid, block>>>(d_data, d_coarse_labels, d_random_proj, N, DIM, NUM_BUCKETS, N_PROJ, N_PROBES);
    cudaDeviceSynchronize();
}
