#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "cosine.h"

__global__ void cosine_similarity(
    const float*    d_samples,
    const float*    d_input,
    float*          d_output,             
    int N, int D) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N)
        return;

    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for(int d = 0; d < D; d++)
    {
        float a = d_samples[idx * D + d];
        float b = d_input[d];
        dot += a * b;
        normA += a * a;
        normB += b * b;
    }

    d_output[idx] = dot / (sqrtf(normA) * sqrtf(normB));
}

void launch_cosine_similarity(
    const float*    d_samples,
    const float*    d_input,
    float*          d_output,
    int N, int D, int TPB,
    cudaStream_t stream)
{
    dim3 blocks = (N + TPB - 1) / TPB;
    dim3 threads(TPB);
    cosine_similarity<<<blocks, threads, 0, stream>>>(
        d_samples,
        d_input,
        d_output,
        N, D);
}