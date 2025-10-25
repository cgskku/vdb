#include <cuda_runtime.h>
#include <math.h>
#include "../include/kmeans_bank.h"

// Distance calculation function
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

__global__ void kmeans_update_centers_tile_kernel(float *d_samples, int *d_clusterIndices, 
                                                 float *d_clusterCenters, int *d_clusterSizes, 
                                                 int N, int K, int DIM, int tile_start, int tile_size)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = tile_start + globalThreadIndex;
    
    if (localIndex >= N || globalThreadIndex >= tile_size) return;

    int cluster_id = d_clusterIndices[localIndex];

    // Accumulate the coordinates of each data point to the cluster centroid it belongs to
    for (int d = 0; d < DIM; ++d)
    {
        atomicAdd(&d_clusterCenters[cluster_id * DIM + d], d_samples[localIndex * DIM + d]);
    }

    // Increase the count of data points included in the cluster the current data point belongs to by 1
    atomicAdd(&d_clusterSizes[cluster_id], 1);
}

__global__ void kmeans_average_centers_kernel(float *d_clusterCenters, int *d_clusterSizes, int K, int DIM)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= K) return;

    // Calculate the new centroid of each cluster by computing the average coordinates 
    for (int d = 0; d < DIM; ++d)
    {
        if (d_clusterSizes[globalThreadIndex] > 0) {
            d_clusterCenters[globalThreadIndex * DIM + d] /= d_clusterSizes[globalThreadIndex];
        }
    }
}

__global__ void kmeans_labeling_tile_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, 
                                           int N, int K, int DIM, int tile_start, int tile_size)
{
    extern __shared__ float shared_centroids[];
    
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIndex = tile_start + globalThreadIndex;
    const int tid = threadIdx.x;
    
    if (localIndex >= N || globalThreadIndex >= tile_size) return;

    float minDistance = INFINITY;
    int closestCenterIndex = -1;

    const float* curPoint = &d_samples[localIndex * DIM];

    // bank-aligned padding for Bank alignment (32의 배수)
    const int BANK_SIZE = 32;
    const int PADDED_DIM = ((DIM + BANK_SIZE - 1) / BANK_SIZE) * BANK_SIZE;
    
    // centroids are processed in chunks, limited by shared memory size
    const int MAX_CENTROIDS_PER_CHUNK = 8; // shared memory size according to shared memory size
    int centroids_per_chunk = min(MAX_CENTROIDS_PER_CHUNK, K);
    
    for (int chunk_start = 0; chunk_start < K; chunk_start += centroids_per_chunk)
    {
        int chunk_end = min(chunk_start + centroids_per_chunk, K);
        
        // load centroids to shared memory (bank-aligned)
        for (int k = chunk_start; k < chunk_end; k++)
        {
            for (int d = tid; d < DIM; d += blockDim.x)
            {
                int local_k = k - chunk_start;
                shared_centroids[local_k * PADDED_DIM + d] = d_clusterCenters[k * DIM + d];
            }
        }
        __syncthreads();
        
        // calculate distances from shared memory
        for (int k = chunk_start; k < chunk_end; k++)
        {
            int local_k = k - chunk_start;
            float distance = 0.0f;
            
            for (int d = 0; d < DIM; d++)
            {
                float diff = curPoint[d] - shared_centroids[local_k * PADDED_DIM + d];
                distance += diff * diff;
            }
            distance = sqrt(distance);
            
            if (distance < minDistance)
            {
                minDistance = distance;
                closestCenterIndex = k;
            }
        }
        __syncthreads();
    }

    d_clusterIndices[localIndex] = closestCenterIndex;
}



void launch_kmeans_labeling_tile(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, 
                                int N, int TPB, int K, int DIM, int tile_start, int tile_size)
{
    int actual_tile_size = min(tile_size, N - tile_start);
    if (actual_tile_size <= 0) return;
    
    // bank-aligned,shared memory size calculation
    const int BANK_SIZE = 32;
    const int PADDED_DIM = ((DIM + BANK_SIZE - 1) / BANK_SIZE) * BANK_SIZE;
    const int MAX_CENTROIDS_PER_CHUNK = 8;
    int shared_mem_size = MAX_CENTROIDS_PER_CHUNK * PADDED_DIM * sizeof(float);
    
    kmeans_labeling_tile_kernel<<<(actual_tile_size + TPB - 1) / TPB, TPB, shared_mem_size>>>(
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

void launch_kmeans_average_centers(float *d_clusterCenters, int *d_clusterSizes, int K, int DIM, int TPB)
{
    kmeans_average_centers_kernel<<<(K + TPB - 1) / TPB, TPB>>>(d_clusterCenters, d_clusterSizes, K, DIM);
}

