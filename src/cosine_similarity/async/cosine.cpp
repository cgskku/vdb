#include <iostream>
#include <fstream>
#include <cstdlib>

#include <cuda_runtime.h>
#include "cosine.h"

int main(int argc, char *argv[])
{
    std::cout.precision(10);
    std::size_t N = 1000000; // Number of data points
    std::size_t dimension = 1536; // Dimension of data points

    int TPB = 128;
    int fixed_mini_batches_size = 100000;
    int num_mini_batches = (N + fixed_mini_batches_size - 1) / fixed_mini_batches_size;

    std::vector<float> h_samples(N * dimension);
    std::vector<float> h_output(N);

    // Generate data
    generate_sample_data(h_samples, N, dimension);
    std::cout << "Kernel Start" << std::endl;
    auto total_kernel_start = std::chrono::high_resolution_clock::now();

    // Create CUDA streams
    std::vector<cudaStream_t> stream(num_mini_batches);
    for(int i = 0; i < num_mini_batches; i++){
        cudaStreamCreate(&stream[i]);
    }

    // Allocate input data on device input은 h_samples의 첫 번째 벡터
    float* d_input = nullptr;
    cudaMalloc(&d_input, dimension * sizeof(float));
    cudaMemcpyAsync(d_input, h_samples.data(), dimension * sizeof(float), cudaMemcpyHostToDevice, stream[0]);

    for(int batch_index = 0; batch_index < num_mini_batches; batch_index++){
        std::size_t start_index = batch_index * fixed_mini_batches_size;
        std::size_t end_index = std::min(start_index + fixed_mini_batches_size, N);
        std::size_t batch_size = end_index - start_index;

        // Allocate device memory
        float *d_samples = nullptr, *d_output = nullptr;
        cudaMallocAsync((void**)&d_samples, batch_size * dimension * sizeof(float), stream[batch_index]);
        cudaMallocAsync((void**)&d_output, batch_size * sizeof(float), stream[batch_index]);

        // Copy data to device
        const float *h_samples_ptr = h_samples.data() + start_index * dimension;
        cudaMemcpyAsync(d_samples, h_samples_ptr, batch_size * dimension * sizeof(float), cudaMemcpyHostToDevice, stream[batch_index]);

        // Launch kernel
        launch_cosine_similarity(d_samples, d_input, d_output, batch_size, dimension, TPB, stream[batch_index]);

        // Copy result back to host
        float *h_output_ptr = h_output.data() + start_index;
        cudaMemcpyAsync(h_output_ptr, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream[batch_index]);

        // free device memory
        cudaFreeAsync(d_samples, stream[batch_index]);
        cudaFreeAsync(d_output, stream[batch_index]);
    }

    for(int i = 0; i < num_mini_batches; i++){
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    auto total_kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_kernel_elapsed = total_kernel_end - total_kernel_start;
    std::cout << "Async Kernel time: " << total_kernel_elapsed.count() << " ms" << std::endl;

    std::cout << "[0] Cosine similarity result vs all:\n";
    for(int i = 0; i < std::min(N, (size_t)10); i++){
        std::cout << "sim[0][" << i << "] = " << h_output[i] << "\n";
    }

    cudaFree(d_input);
    return 0;
}