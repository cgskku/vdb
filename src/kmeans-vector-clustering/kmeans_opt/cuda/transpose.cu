#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" 
{

__global__ void transpose_kernel(float* src, float* dst, int rows, int cols) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) dst[x * rows + y] = src[y * cols + x];
}

void transpose_data(float* src, float* dst, int rows, int cols, cudaStream_t stream) 
{
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    transpose_kernel<<<grid, block, 0, stream>>>(src, dst, rows, cols);
}

}
