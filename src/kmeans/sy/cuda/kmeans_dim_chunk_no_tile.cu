#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "../include/kmeans.h"

__device__ float get_chunk_distance(const float* point, const float* partialCentroid, int chunkSize, int offset) 
{
    float squaredDistance = 0.0f;
    for (int i = 0; i < chunkSize; ++i) 
    {
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

    // @@
    const float* curPointFull = &d_samples[globalThreadIndex * DIM];

    // no tiling
    for (int k = 0; k < K; ++k) {
        float totalDistance = 0.0f;

        for (int offset = 0; offset < DIM; offset += chunkSize) 
        {
            const int curChunkSize = min(chunkSize, DIM - offset);
            int i4 = (tid << 2);
            const float* centBase = &d_clusterCenters[k * DIM + offset];
            for (int idx = i4; idx < curChunkSize; idx += (blockDim.x << 2)) 
            {
                if (idx + 3 < curChunkSize) 
                {
                    float4 v = reinterpret_cast<const float4*>(centBase)[idx >> 2];
                    reinterpret_cast<float4*>(sharedCentroids)[idx >> 2] = v;
                } 
                else 
                {
                    for (int t = 0; t < 4 && idx + t < curChunkSize; ++t) sharedCentroids[idx + t] = centBase[idx + t];
                }
            }
            __syncthreads();

            const float* curPoint = curPointFull + offset;
            float partial = 0.0f;

            int d4 = (curChunkSize & ~3);
            for (int j = 0; j < d4; j += 4) 
            {
                float4 p = reinterpret_cast<const float4*>(curPoint)[j >> 2];
                float4 c = reinterpret_cast<const float4*>(sharedCentroids)[j >> 2];
                float dx0 = p.x - c.x; partial += dx0 * dx0;
                float dx1 = p.y - c.y; partial += dx1 * dx1;
                float dx2 = p.z - c.z; partial += dx2 * dx2;
                float dx3 = p.w - c.w; partial += dx3 * dx3;
            }
            for (int j = d4; j < curChunkSize; ++j) 
            {
                float diff = curPoint[j] - sharedCentroids[j];
                partial += diff * diff;
            }

            totalDistance += partial;
            __syncthreads(); 

            if (totalDistance >= minDistance) 
            {
                for (; offset + chunkSize < DIM; offset += chunkSize) {}
                break;
            }
        }

        if (totalDistance < minDistance) 
        {
            minDistance = totalDistance;
            closestCenterIndex = k;
        }
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

        for(int i = tid; i < K * curChunkSize; i += blockDim.x) sharedPsum[i] = 0.0f;
        __syncthreads();

        if (globalThreadIndex < N) 
        {
            int cluster_id = d_clusterIndices[globalThreadIndex];
            const float* curPoint = &d_samples[globalThreadIndex * DIM + offset];
            for (int d = 0; d < curChunkSize; ++d) atomicAdd(&sharedPsum[cluster_id * curChunkSize + d], curPoint[d]);
        }
        __syncthreads();

        for (int i = tid; i < K * curChunkSize; i += blockDim.x) 
        {
            int c = i / curChunkSize;       
            int d = i % curChunkSize;      
            atomicAdd(&d_clusterCenters[c * DIM + offset + d], sharedPsum[i]);
        }
        __syncthreads(); 
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
    int shared_memory_size = chunkSize * sizeof(float); // Use chunkSize from host

    dim3 block(TPB);
    dim3 grid((N + TPB - 1) / TPB);
    kmeans_labeling_kernel<<<grid, block, shared_memory_size>>>(d_samples, d_clusterIndices, d_clusterCenters, N, K, DIM, chunkSize);
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