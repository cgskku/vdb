#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cublas_v2.h>
#include "include/kmeans.h"

__device__ float get_distance(const float* point, const float* centroid, int DIM)
{
    float squaredDistance = 0.0f;
    for (int d = 0; d < DIM; ++d)
    {
        float vecDiff = point[d] - centroid[d];
        squaredDistance += vecDiff * vecDiff;
    }
    return sqrt(squaredDistance);
}

__global__ void kmeans_labeling_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int N, int K, int DIM)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= N) return;

    float minDistance = INFINITY;
    int closestCenterIndex = -1;

    const float* curPoint = &d_samples[globalThreadIndex * DIM];

    for (int k = 0; k < K; ++k)
    {
        const float* curCentroid = &d_clusterCenters[k * DIM];
        float distance = get_distance(curPoint, curCentroid, DIM);
        if (distance < minDistance)
        {
            minDistance = distance;
            closestCenterIndex = k;
        }
    }

    d_clusterIndices[globalThreadIndex] = closestCenterIndex;
}

__global__ void kmeans_update_centers_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int K, int DIM)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= N) return;

    int cluster_id = d_clusterIndices[globalThreadIndex];

    // accumulate the coordinates of each data point to the cluster centroid it belongs to
    for (int d = 0; d < DIM; ++d)
    {
        atomicAdd(&d_clusterCenters[cluster_id * DIM + d], d_samples[globalThreadIndex * DIM + d]);
    }

    atomicAdd(&d_clusterSizes[cluster_id], 1);
}

__global__ void kmeans_average_centers_kernel(float *d_clusterCenters, int *d_clusterSizes, int K, int DIM)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= K) return;

    // calculate the new centroid of each cluster by computing the average coordinates 
    for (int d = 0; d < DIM; ++d)
    {
        if (d_clusterSizes[globalThreadIndex] > 0) {
            d_clusterCenters[globalThreadIndex * DIM + d] /= d_clusterSizes[globalThreadIndex];
        }
    }
}

void launch_kmeans_labeling(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int N, int TPB, int K, int DIM)
{
    kmeans_labeling_kernel<<<(N + TPB - 1) / TPB, TPB>>>(d_samples, d_clusterIndices, d_clusterCenters, N, K, DIM);
}

void launch_kmeans_update_center(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int TPB, int K, int DIM)
{
    cudaMemset(d_clusterCenters, 0, K * DIM * sizeof(float));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));

    kmeans_update_centers_kernel<<<(N + TPB - 1) / TPB, TPB>>>(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM);
    kmeans_average_centers_kernel<<<(K + TPB - 1) / TPB, TPB>>>(d_clusterCenters, d_clusterSizes, K, DIM);
}

void launch_kmeans_average_centers(float *d_clusterCenters, int *d_clusterSizes, int K, int DIM, int TPB)
{
    kmeans_average_centers_kernel<<<(K + TPB - 1) / TPB, TPB>>>(d_clusterCenters, d_clusterSizes, K, DIM);
}

// tiling

__global__ void kmeans_labeling_chunk_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int N, int K, int DIM, int chunkSize)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= N) return;

    float minDistance = INFINITY;
    int closestCenterIndex = -1;

    const float* curPoint = &d_samples[globalThreadIndex * DIM];

    // process centroids in chunks 
    for (int chunk = 0; chunk < (K + chunkSize - 1) / chunkSize; ++chunk)
    {
        int startK = chunk * chunkSize;
        int endK = min(startK + chunkSize, K);
        
        for (int k = startK; k < endK; ++k)
        {
            const float* curCentroid = &d_clusterCenters[k * DIM];
            float distance = get_distance(curPoint, curCentroid, DIM);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestCenterIndex = k;
            }
        }
    }

    d_clusterIndices[globalThreadIndex] = closestCenterIndex;
}

__global__ void kmeans_update_centers_chunk_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int K, int DIM, int chunkSize)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= N) return;

    int cluster_id = d_clusterIndices[globalThreadIndex];

    // process dimensions in chunks
    for (int chunk = 0; chunk < (DIM + chunkSize - 1) / chunkSize; ++chunk)
    {
        int startD = chunk * chunkSize;
        int endD = min(startD + chunkSize, DIM);
        
        for (int d = startD; d < endD; ++d)
        {
            atomicAdd(&d_clusterCenters[cluster_id * DIM + d], d_samples[globalThreadIndex * DIM + d]);
        }
    }

    atomicAdd(&d_clusterSizes[cluster_id], 1);
}

__global__ void kmeans_average_centers_chunk_kernel(float *d_clusterCenters, int *d_clusterSizes, int K, int DIM, int chunkSize)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= K) return;

    // process dimensions in chunks
    for (int chunk = 0; chunk < (DIM + chunkSize - 1) / chunkSize; ++chunk)
    {
        int startD = chunk * chunkSize;
        int endD = min(startD + chunkSize, DIM);
        
        for (int d = startD; d < endD; ++d)
        {
            if (d_clusterSizes[globalThreadIndex] > 0) {
                d_clusterCenters[globalThreadIndex * DIM + d] /= d_clusterSizes[globalThreadIndex];
            }
        }
    }
}

void launch_kmeans_labeling_chunk(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int N, int TPB, int K, int DIM, int chunkSize)
{
    kmeans_labeling_chunk_kernel<<<(N + TPB - 1) / TPB, TPB>>>(d_samples, d_clusterIndices, d_clusterCenters, N, K, DIM, chunkSize);
}

void launch_kmeans_update_center_chunk(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int TPB, int K, int DIM, int chunkSize)
{
    cudaMemset(d_clusterCenters, 0, K * DIM * sizeof(float));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));

    kmeans_update_centers_chunk_kernel<<<(N + TPB - 1) / TPB, TPB>>>(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM, chunkSize);
    kmeans_average_centers_chunk_kernel<<<(K + TPB - 1) / TPB, TPB>>>(d_clusterCenters, d_clusterSizes, K, DIM, chunkSize);
}


__global__ void kmeans_labeling_tile_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, 
                                           int N, int K, int DIM, int tile_start, int tile_size)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = tile_start + globalThreadIndex;
    
    if (localIndex >= N || globalThreadIndex >= tile_size) return;

    float minDistance = INFINITY;
    int closestCenterIndex = -1;

    const float* curPoint = &d_samples[localIndex * DIM];

    for (int k = 0; k < K; ++k)
    {
        const float* curCentroid = &d_clusterCenters[k * DIM];
        float distance = get_distance(curPoint, curCentroid, DIM);
        if (distance < minDistance)
        {
            minDistance = distance;
            closestCenterIndex = k;
        }
    }

    d_clusterIndices[localIndex] = closestCenterIndex;
}

__global__ void kmeans_update_centers_tile_kernel(float *d_samples, int *d_clusterIndices, 
                                                 float *d_clusterCenters, int *d_clusterSizes, 
                                                 int N, int K, int DIM, int tile_start, int tile_size)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = tile_start + globalThreadIndex;
    
    if (localIndex >= N || globalThreadIndex >= tile_size) return;

    int cluster_id = d_clusterIndices[localIndex];

    for (int d = 0; d < DIM; ++d)
    {
        atomicAdd(&d_clusterCenters[cluster_id * DIM + d], d_samples[localIndex * DIM + d]);
    }

    atomicAdd(&d_clusterSizes[cluster_id], 1);
}

void launch_kmeans_labeling_tile(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, 
                                int N, int TPB, int K, int DIM, int tile_start, int tile_size)
{
    int actual_tile_size = min(tile_size, N - tile_start);
    if (actual_tile_size <= 0) return;
    
    kmeans_labeling_tile_kernel<<<(actual_tile_size + TPB - 1) / TPB, TPB>>>(
        d_samples, d_clusterIndices, d_clusterCenters, N, K, DIM, tile_start, actual_tile_size);
}

void launch_kmeans_update_center_tile(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, 
                                     int *d_clusterSizes, int N, int TPB, int K, int DIM, 
                                     int tile_start, int tile_size)
{
    int actual_tile_size = min(tile_size, N - tile_start);
    if (actual_tile_size <= 0) return;
    
    kmeans_update_centers_tile_kernel<<<(actual_tile_size + TPB - 1) / TPB, TPB>>>(
        d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM, tile_start, actual_tile_size);
}

// corner turning kernels
__global__ void transpose_centers_kernel(float *d_centroids, float *d_centroids_T, int K, int DIM)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (k < K && d < DIM) 
    {
        d_centroids_T[d * K + k] = d_centroids[k * DIM + d];
    }
}

__global__ void kmeans_labeling_corner_turning_kernel(float *d_samples, int *d_clusterIndices, 
                                                     float *d_centroids_T, int N, int K, int DIM)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= N) return;

    float minDistance = INFINITY;
    int closestCenterIndex = -1;

    const float* curPoint = &d_samples[globalThreadIndex * DIM];

    for (int k = 0; k < K; ++k)
    {
        float distance = 0.0f;
        for (int d = 0; d < DIM; ++d)
        {
            float vecDiff = curPoint[d] - d_centroids_T[d * K + k];
            distance += vecDiff * vecDiff;
        }
        
        if (distance < minDistance)
        {
            minDistance = distance;
            closestCenterIndex = k;
        }
    }
    d_clusterIndices[globalThreadIndex] = closestCenterIndex;
}

__global__ void kmeans_labeling_corner_turning_tile_kernel(float *d_samples, int *d_clusterIndices, 
                                                          float *d_centroids_T, int N, int K, int DIM, 
                                                          int tile_start, int tile_size)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = tile_start + globalThreadIndex;
    
    if (localIndex >= N || globalThreadIndex >= tile_size) return;

    float minDistance = INFINITY;
    int closestCenterIndex = -1;

    const float* curPoint = &d_samples[localIndex * DIM];

    for (int k = 0; k < K; ++k)
    {
        float distance = 0.0f;
        for (int d = 0; d < DIM; ++d)
        {
            float vecDiff = curPoint[d] - d_centroids_T[d * K + k];
            distance += vecDiff * vecDiff;
        }
        
        if (distance < minDistance)
        {
            minDistance = distance;
            closestCenterIndex = k;
        }
    }

    d_clusterIndices[localIndex] = closestCenterIndex;
}

// corner turning functions
void transpose_centers(float *d_centroids, float *d_centroids_T, int K, int DIM, int TPB)
{
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (DIM + block.y - 1) / block.y);
    transpose_centers_kernel<<<grid, block>>>(d_centroids, d_centroids_T, K, DIM);
}

void launch_kmeans_labeling_corner_turning(float *d_samples, int *d_clusterIndices, 
                                          float *d_centroids_T, int N, int TPB, int K, int DIM)
{
    kmeans_labeling_corner_turning_kernel<<<(N + TPB - 1) / TPB, TPB>>>(
        d_samples, d_clusterIndices, d_centroids_T, N, K, DIM);
}

void launch_kmeans_labeling_corner_turning_tile(float *d_samples, int *d_clusterIndices, 
                                               float *d_centroids_T, int N, int TPB, int K, int DIM, 
                                               int tile_start, int tile_size)
{
    int actual_tile_size = min(tile_size, N - tile_start);
    if (actual_tile_size <= 0) return;
    
    kmeans_labeling_corner_turning_tile_kernel<<<(actual_tile_size + TPB - 1) / TPB, TPB>>>(
        d_samples, d_clusterIndices, d_centroids_T, N, K, DIM, tile_start, actual_tile_size);
}

// stream-based functions

void launch_kmeans_labeling_corner_turning_tile_stream(float *d_samples, int *d_clusterIndices, 
                                                      float *d_centroids_T, int N, int TPB, int K, int DIM, 
                                                      int tile_start, int tile_size, cudaStream_t stream)
{
    int actual_tile_size = min(tile_size, N - tile_start);
    if (actual_tile_size <= 0) return;
    
    kmeans_labeling_corner_turning_tile_kernel<<<(actual_tile_size + TPB - 1) / TPB, TPB, 0, stream>>>(
        d_samples, d_clusterIndices, d_centroids_T, N, K, DIM, tile_start, actual_tile_size);
}

void launch_kmeans_labeling_tile_stream(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, 
                                       int N, int TPB, int K, int DIM, int tile_start, int tile_size, cudaStream_t stream)
{
    int actual_tile_size = min(tile_size, N - tile_start);
    if (actual_tile_size <= 0) return;
    
    kmeans_labeling_tile_kernel<<<(actual_tile_size + TPB - 1) / TPB, TPB, 0, stream>>>(
        d_samples, d_clusterIndices, d_clusterCenters, N, K, DIM, tile_start, actual_tile_size);
}

// for cuBLAS
__global__ void add_norms_to_distances_kernel(float* d_datapoints, float* d_centroids, 
                                            float* d_distances, int N, int K, int DIM)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    
    int n = idx / K;
    int k = idx % K;
    
    float data_norm = 0.0f;
    float centroid_norm = 0.0f;
    
    for (int d = 0; d < DIM; ++d) {
        float data_val = d_datapoints[n * DIM + d];
        float centroid_val = d_centroids[k * DIM + d];
        data_norm += data_val * data_val;
        centroid_norm += centroid_val * centroid_val;
    }
    
    d_distances[idx] += data_norm + centroid_norm;
}

__global__ void find_min_distance_kernel(float* d_distances, int* d_assign, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    float min_dist = d_distances[n * K];
    int min_k = 0;
    
    for (int k = 1; k < K; ++k) {
        float dist = d_distances[n * K + k];
        if (dist < min_dist) {
            min_dist = dist;
            min_k = k;
        }
    }
    
    d_assign[n] = min_k;
}

void launch_kmeans_labeling_cublas(float* d_datapoints, int* d_assign, float* d_centroids, 
                                  float* d_distances, int N, int K, int DIM, cublasHandle_t handle) 
{
    const float alpha = -2.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, K, DIM,
                &alpha, d_datapoints, DIM,
                d_centroids, DIM,          
                &beta, d_distances, N);
    
    add_norms_to_distances_kernel<<<(N * K + 255) / 256, 256>>>(
        d_datapoints, d_centroids, d_distances, N, K, DIM);
    
    find_min_distance_kernel<<<(N + 255) / 256, 256>>>(
        d_distances, d_assign, N, K);
}

void launch_kmeans_update_center_tile_stream(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, 
                                            int *d_clusterSizes, int N, int TPB, int K, int DIM, 
                                            int tile_start, int tile_size, cudaStream_t stream)
{
    int actual_tile_size = min(tile_size, N - tile_start);
    if (actual_tile_size <= 0) return;
    
    kmeans_update_centers_tile_kernel<<<(actual_tile_size + TPB - 1) / TPB, TPB, 0, stream>>>(
        d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM, tile_start, actual_tile_size);
}

void launch_kmeans_average_centers_stream(float *d_clusterCenters, int *d_clusterSizes, int K, int DIM, int TPB, cudaStream_t stream)
{
    kmeans_average_centers_kernel<<<(K + TPB - 1) / TPB, TPB, 0, stream>>>(d_clusterCenters, d_clusterSizes, K, DIM);
}

void transpose_centers_stream(float *d_centroids, float *d_centroids_T, int K, int DIM, int TPB, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (DIM + block.y - 1) / block.y);
    transpose_centers_kernel<<<grid, block, 0, stream>>>(d_centroids, d_centroids_T, K, DIM);
}
