#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "kmeans.h"

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
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global index of each thread, representing the position of the data point in the array
    if (globalThreadIndex >= N) return;

    float minDistance = INFINITY;
    int closestCenterIndex = -1;

    const float* curPoint = &d_samples[globalThreadIndex * DIM];

    // Iterate through cluster centroids to calculate distance
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

    // Accumulate the coordinates of each data point to the cluster centroid it belongs to
    for (int d = 0; d < DIM; ++d)
    {
        atomicAdd(&d_clusterCenters[cluster_id * DIM + d], d_samples[globalThreadIndex * DIM + d]);
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
