// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <math.h>
// #include "../include/kmeans.h"

// __device__ float get_chunk_distance(const float* point, const float* partialCentroid, int chunkSize, int offset) 
// {
//     float squaredDistance = 0.0f;
//     for(int i = 0; i < chunkSize; ++i) 
//     {
//         float vecDiff = point[i] - partialCentroid[i];
//         squaredDistance += vecDiff * vecDiff;
//     }
//     return squaredDistance;
// }

// __global__ void kmeans_labeling_kernel(
//     float* d_samples,
//     int* d_clusterIndices,
//     float* d_clusterCenters,
//     int N, int K, int DIM, int chunkSize) 
// {
//     extern __shared__ float shardMemory[]; // Shared memory for centroids
//     float* sharedCentroids = shardMemory;

//     const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     if (globalThreadIndex >= N) return;


//     const int tid = threadIdx.x;

//     float distances[32]; // Assume K <= 32 for simplicity
//     for (int k = 0; k < K; ++k) {
//         distances[k] = 0.0f;
//     }

//     // Loop through DIM chunks
//     for (int offset = 0; offset < DIM; offset += chunkSize) 
//     {
//         int curChunkSize = min(chunkSize, DIM - offset); 

//         // Load centroids into shared memory for this chunk
//         for (int i = tid; i < K * curChunkSize; i += blockDim.x) 
//         {
//             int c = i / curChunkSize;    
//             int d = i % curChunkSize;       
//             sharedCentroids[c * curChunkSize + d] = d_clusterCenters[c * DIM + offset + d];
//         }
//         __syncthreads(); // Ensure all centroids are loaded

//         const float* curPoint = &d_samples[globalThreadIndex * DIM + offset];
//         for (int k = 0; k < K; ++k) {
//             const float* partialCentroid = &sharedCentroids[k * curChunkSize];
//             float chunkDistance = get_chunk_distance(curPoint, partialCentroid, curChunkSize, 0);
//             distances[k] += chunkDistance;
//         }
//         __syncthreads();
//     }

//     // Find minimum distance
//     float minDistance = distances[0];
//     int closestCenterIndex = 0;
//     for (int k = 1; k < K; ++k) {
//         if (distances[k] < minDistance) {
//             minDistance = distances[k];
//             closestCenterIndex = k;
//         }
//     }

//     d_clusterIndices[globalThreadIndex] = closestCenterIndex;
// }

// __global__ void kmeans_update_centers_kernel(
//     float* d_samples,
//     int* d_clusterIndices,
//     float* d_clusterCenters,
//     int* d_clusterSizes,
//     int N, int K, int DIM, int chunkSize) 
// {
//     extern __shared__ float shardMemory[];  // K * chunkSize
//     float* sharedPsum = shardMemory;

//     const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     const int tid = threadIdx.x;

//     for (int offset = 0; offset < DIM; offset += chunkSize) {
//         int curChunkSize = min(chunkSize, DIM - offset);

//         for(int i = tid; i < K * curChunkSize; i += blockDim.x) {
//             sharedPsum[i] = 0.0f;
//         }
//         __syncthreads();

//         if (globalThreadIndex < N) {
//             int cluster_id = d_clusterIndices[globalThreadIndex];
//             const float* curPoint = &d_samples[globalThreadIndex * DIM + offset];
//             for (int d = 0; d < curChunkSize; ++d) {
//                 atomicAdd(&sharedPsum[cluster_id * curChunkSize + d], curPoint[d]);
//             }
//         }
//         __syncthreads();

//         for (int i = tid; i < K * curChunkSize; i += blockDim.x) {
//             int c = i / curChunkSize;       
//             int d = i % curChunkSize;      
//             atomicAdd(&d_clusterCenters[c * DIM + offset + d], sharedPsum[i]);
//         }
//     }

//     if (globalThreadIndex < N) {
//         int cluster_id = d_clusterIndices[globalThreadIndex];
//         atomicAdd(&d_clusterSizes[cluster_id], 1);
//     }
// }

// __global__ void kmeans_average_centers_kernel(float* d_clusterCenters, int* d_clusterSizes, int K, int DIM) {
//     const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     if (globalThreadIndex >= K) return;

//     for (int d = 0; d < DIM; ++d) {
//         if (d_clusterSizes[globalThreadIndex] > 0) {
//             d_clusterCenters[globalThreadIndex * DIM + d] /= d_clusterSizes[globalThreadIndex];
//         }
//     }
// }

// void launch_kmeans_labeling_chunk(
//     float* d_samples, int* d_clusterIndices, 
//     float* d_clusterCenters, 
//     int N, int TPB, int K, int DIM, int chunkSize) 
// {
//     int shared_memory_size = K * chunkSize * sizeof(float); // Use chunkSize from host

//     dim3 block(TPB);
//     dim3 grid((N + TPB - 1) / TPB);
//     kmeans_labeling_kernel<<<grid, block, shared_memory_size>>>(d_samples, d_clusterIndices, d_clusterCenters, N, K, DIM, chunkSize);
// }

// void launch_kmeans_update_center_chunk(
//     float* d_samples, int* d_clusterIndices, 
//     float* d_clusterCenters, int* d_clusterSizes, 
//     int N, int TPB, int K, int DIM, int chunkSize
//     )
// {
//     cudaMemset(d_clusterCenters, 0, K * DIM * sizeof(float));
//     cudaMemset(d_clusterSizes, 0, K * sizeof(int));
//     int shared_memory_size = K * chunkSize * sizeof(float); // Use chunkSize from host

//     kmeans_update_centers_kernel<<<(N + TPB - 1) / TPB, TPB, shared_memory_size>>>(
//         d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM, chunkSize);
//     cudaDeviceSynchronize();
//     kmeans_average_centers_kernel<<<(K + TPB - 1) / TPB, TPB>>>(d_clusterCenters, d_clusterSizes, K, DIM);
//     cudaDeviceSynchronize();
// }

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "kmeans.h"

__device__ float get_chunk_distance(const float* point, const float* partialCentroid, int chunkSize, int offset) {
    float squaredDistance = 0.0f;
    for (int i = 0; i < chunkSize; ++i) {
        float vecDiff = point[offset + i] - partialCentroid[i];
        squaredDistance += vecDiff * vecDiff;
    }
    return squaredDistance;
}

__global__ void kmeans_labeling_kernel(
    float* d_samples,
    int* d_clusterIndices,
    float* d_clusterCenters,
    int N, int K, int DIM, int chunkSize) 
{
    extern __shared__ float shardMemory[]; // Shared memory for centroids
    float* sharedCentroids = shardMemory;

    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= N) return;

    const int tid = threadIdx.x;

    float minDistance = INFINITY;
    int closestCenterIndex = 0;

    // Loop through DIM chunks
    for (int offset = 0; offset < DIM; offset += chunkSize) {
        int curChunkSize = min(chunkSize, DIM - offset); 

        // Load centroids into shared memory for this chunk
        for (int i = tid; i < K * curChunkSize; i += blockDim.x) {
            int c = i / curChunkSize;    
            int d = i % curChunkSize;       
            sharedCentroids[c * curChunkSize + d] = d_clusterCenters[c * DIM + offset + d];
        }
        __syncthreads(); // Ensure all centroids are loaded

        const float* curPoint = &d_samples[globalThreadIndex * DIM];
        for (int k = 0; k < K; ++k) {
            const float* partialCentroid = &sharedCentroids[k * curChunkSize];
            float squaredDistance = get_chunk_distance(curPoint, partialCentroid, curChunkSize, offset);
            if (squaredDistance < minDistance) {
                minDistance = squaredDistance;
                closestCenterIndex = k;
            }
        }
        __syncthreads();
    }

    d_clusterIndices[globalThreadIndex] = closestCenterIndex;
}

__global__ void kmeans_update_centers_kernel(
    float* d_samples,
    int* d_clusterIndices,
    float* d_clusterCenters,
    int* d_clusterSizes,
    int N, int K, int DIM, int chunkSize) 
{
    extern __shared__ float shardMemory[];  // K * chunkSize
    float* sharedPsum = shardMemory;

    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    for (int offset = 0; offset < DIM; offset += chunkSize) {
        int curChunkSize = min(chunkSize, DIM - offset);

        for(int i = tid; i < K * curChunkSize; i += blockDim.x) {
            sharedPsum[i] = 0.0f;
        }
        __syncthreads();

        if (globalThreadIndex < N) {
            int cluster_id = d_clusterIndices[globalThreadIndex];
            const float* curPoint = &d_samples[globalThreadIndex * DIM + offset];
            for (int d = 0; d < curChunkSize; ++d) {
                atomicAdd(&sharedPsum[cluster_id * curChunkSize + d], curPoint[d]);
            }
        }
        __syncthreads();

        for (int i = tid; i < K * curChunkSize; i += blockDim.x) {
            int c = i / curChunkSize;       
            int d = i % curChunkSize;      
            atomicAdd(&d_clusterCenters[c * DIM + offset + d], sharedPsum[i]);
        }
    }

    if (globalThreadIndex < N) {
        int cluster_id = d_clusterIndices[globalThreadIndex];
        atomicAdd(&d_clusterSizes[cluster_id], 1);
    }
}

__global__ void kmeans_average_centers_kernel(float* d_clusterCenters, int* d_clusterSizes, int K, int DIM) {
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= K) return;

    for (int d = 0; d < DIM; ++d) {
        if (d_clusterSizes[globalThreadIndex] > 0) {
            d_clusterCenters[globalThreadIndex * DIM + d] /= d_clusterSizes[globalThreadIndex];
        }
    }
}

void launch_kmeans_labeling_chunk(
    float* d_samples, int* d_clusterIndices, 
    float* d_clusterCenters, 
    int N, int TPB, int K, int DIM, int chunkSize) 
{
    int shared_memory_size = K * chunkSize * sizeof(float); // Use chunkSize from host

    kmeans_labeling_kernel<<<(N + TPB - 1) / TPB, TPB, shared_memory_size>>>(
        d_samples, d_clusterIndices, d_clusterCenters, N, K, DIM, chunkSize);
}

void launch_kmeans_update_center_chunk(
    float* d_samples, int* d_clusterIndices, 
    float* d_clusterCenters, int* d_clusterSizes, 
    int N, int TPB, int K, int DIM, int chunkSize
    )
{
    cudaMemset(d_clusterCenters, 0, K * DIM * sizeof(float));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));
    int shared_memory_size = K * chunkSize * sizeof(float); // Use chunkSize from host

    kmeans_update_centers_kernel<<<(N + TPB - 1) / TPB, TPB, shared_memory_size>>>(
        d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM, chunkSize);
    cudaDeviceSynchronize();
    kmeans_average_centers_kernel<<<(K + TPB - 1) / TPB, TPB>>>(d_clusterCenters, d_clusterSizes, K, DIM);
    cudaDeviceSynchronize();
}