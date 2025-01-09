#include <cuda_runtime.h>
//#include <device_launch_parameter.h>
#include <math.h>
#include "kmeans_bank.h"
#define WARPSIZE 32

__device__ int d_chunkSize = 256;

__global__ void kmeans_labeling_kernel(
    float* d_samples,
    int* d_clusterIndices,
    float* d_clusterCenters,
    int *d_clusterSizes,
    int sharedMemSize,
    int N, int K, int DIM
) {

    extern __shared__ float sharedMemory[];
    float *squaredDistanceMem = &sharedMemory[0];
    //int maxWarp = sharedMemSize / sizeof(float) / WARPSIZE;
    int partialDim = DIM / WARPSIZE;
    //int offsetDim = DIM % WARPSIZE;
    int distanceSize = (partialDim + 1) * WARPSIZE;
    float *sharedCentroids = &sharedMemory[distanceSize];
    //const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    const int tid = threadIdx.x;
    squaredDistanceMem[DIM] = INFINITY;
    int *cluster_id = (int *)&squaredDistanceMem[DIM + 1];

    int maxCentroidSapce = sharedMemSize - (distanceSize * sizeof(float));
    int maxk = maxCentroidSapce / DIM / sizeof(float);
    int k = maxk;
    for (int o = 0; o < K; o += k)
    {
        k = min(maxk, K - o);

        for (int j = tid; j < k * DIM; j += blockDim.x)
        {
            sharedCentroids[j] = d_clusterCenters[o * DIM + j];
        }
        __syncthreads();

        for (int j = 0; j < k; j++)
        {
            for (int i = tid; i < DIM; i += blockDim.x)
            {
                float dataPoint = d_samples[blockIdx.x * DIM + i];
                float kPoint = sharedCentroids[j * DIM + i];
                float diff = dataPoint - kPoint;
                squaredDistanceMem[i] = diff * diff;
            }

            for (int stride = distanceSize / 2; stride > 0; stride /= 2)
            {
                for (int i = tid; i < stride; i+= blockDim.x)
                {
                    if (i + stride < DIM)
                    {
                        squaredDistanceMem[i] += squaredDistanceMem[i + stride];
                    }
                }

                __syncthreads();
            }

            if (tid == 0)
            {
                if (squaredDistanceMem[DIM] > squaredDistanceMem[tid])
                {
                    squaredDistanceMem[DIM] = squaredDistanceMem[tid];
                    *cluster_id = o + j;
                    //d_clusterIndices[blockIdx.x] = o + j;
                }
            }
        }
    }

    if (tid == 0)
    {
        atomicAdd(&d_clusterSizes[*cluster_id], 1);
    }
    d_clusterIndices[blockIdx.x] = *cluster_id;
}

// __global__ void kmeans_update_centers_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int K, int DIM)
// {
//     const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     if (globalThreadIndex >= N) return;

//     int cluster_id = d_clusterIndices[globalThreadIndex];

//     // Accumulate the coordinates of each data point to the cluster centroid it belongs to
//     for (int d = 0; d < DIM; ++d)
//     {
//         atomicAdd(&d_clusterCenters[cluster_id * DIM + d], d_samples[globalThreadIndex * DIM + d]);
//     }

//     // Increase the count of data points included in the cluster the current data point belongs to by 1
//     atomicAdd(&d_clusterSizes[cluster_id], 1);
// }

__global__ void kmeans_update_centers_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int K, int DIM)
{
    extern __shared__ float sharedMemory[];
    float *pointSum = &sharedMemory[0];
    int pointNum = blockIdx.x;
    int cluster_id = d_clusterIndices[pointNum];
    const int tid = threadIdx.x;

    // Accumulate the coordinates of each data point to the cluster centroid it belongs to
    // if (tid == 0)
    // {
    //     atomicAdd(&d_clusterSizes[cluster_id], 1);
    // }

    for (int i = tid; i < DIM; i += blockDim.x)
    {
        pointSum[i] = d_samples[pointNum * DIM + i]; 
    }
    __syncthreads();

    for (int i = tid; i < DIM; i += blockDim.x)
    {
        atomicAdd(&d_clusterCenters[cluster_id * DIM + i], (sharedMemory[i] / d_clusterSizes[cluster_id]));
    }

    __syncthreads();
}

// __global__ void kmeans_update_centers_kernel(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N , int K, int DIM)
// {
//     int pointNum = blockIdx.x;
//     int tid = threadIdx.x;
//     int cluster_id = d_clusterIndices[pointNum];

//     if (tid == 0)
//     {
//         atomicAdd(&d_clusterSizes[cluster_id], 1);
//     }

//     for (int i = tid; i < DIM; i+= blockDim.x)
//     {
//         atomicAdd(&d_clusterCenters[cluster_id * DIM + i], d_samples[pointNum * DIM + i]);
//     }

//     __syncthreads();
// }

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

void launch_kmeans_labeling(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int TPB, int K, int DIM)
{
    int shredMemSize = 8 * sizeof(float) * DIM;
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));
    kmeans_labeling_kernel<<<N, TPB, shredMemSize>>>(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, shredMemSize,  N, K, DIM);
}

void launch_kmeans_update_center(float *d_samples, int *d_clusterIndices, float *d_clusterCenters, int *d_clusterSizes, int N, int TPB, int K, int DIM)
{
    cudaMemset(d_clusterCenters, 0, K * DIM * sizeof(float));
    int sharedMemSize = sizeof(float) * DIM;

    kmeans_update_centers_kernel<<<N, TPB, sharedMemSize>>>(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM);
    //kmeans_update_centers_kernel<<<N, TPB>>>(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, K, DIM);
    //kmeans_average_centers_kernel<<<(K + TPB - 1) / TPB, TPB>>>(d_clusterCenters, d_clusterSizes, K, DIM);
}


