#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include "cosine.h"

#define FileFlag 1

int main(int argc, char *argv[])
{
    std::cout.precision(10);
    std::size_t N = 1000000; // Number of data points
    std::size_t dimension = 1536; // Dimension of data points

    int TPB = 128;
    int fixed_mini_batches_size = 1000;
    int num_mini_batches = (N + fixed_mini_batches_size - 1) / fixed_mini_batches_size;

    std::vector<float> h_samples(N * dimension);
    std::vector<float> h_output(N);

    std::cout << "generate_sample_data" << std::endl;
    generate_sample_data(h_samples.data(), N, dimension);
    std::cout << "Kernel Start" << std::endl;
    auto total_kernel_start = std::chrono::high_resolution_clock::now();

    // Allocate pinned memory for input and output
    float *h_batch_pinned = nullptr;
    float *h_out_pinned = nullptr;
    size_t batch_elems = fixed_mini_batches_size * dimension;
    cudaHostAlloc((void**)&h_batch_pinned, batch_elems * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_out_pinned, fixed_mini_batches_size * sizeof(float), cudaHostAllocDefault);

    // Create CUDA stream
    std::vector<cudaStream_t> stream(num_mini_batches);
    for(int i = 0; i < num_mini_batches; i++){
        cudaStreamCreate(&stream[i]);
    }

    std::vector<float*> d_batches(num_mini_batches), d_outputs(num_mini_batches);
    for(int i = 0; i < num_mini_batches; ++i){
        cudaMalloc(&d_batches[i], fixed_mini_batches_size * dimension * sizeof(float));
        cudaMalloc(&d_outputs[i], fixed_mini_batches_size * sizeof(float));
    }

    // Allocate input data on device input은 h_samples의 첫 번째 벡터
    float* d_input = nullptr;
    cudaMalloc(&d_input, dimension * sizeof(float));
    cudaMemcpyAsync(d_input, &h_samples[0], dimension * sizeof(float), cudaMemcpyHostToDevice, stream[0]);

    for(int i = 0; i < num_mini_batches; i++){
        std::size_t start_index = i * fixed_mini_batches_size;
        std::size_t end_index = std::min(start_index + fixed_mini_batches_size, N);
        std::size_t batch_size = end_index - start_index;

        // pageable -> pinned
        std::memcpy(h_batch_pinned, h_samples.data() + start_index * dimension, batch_size * dimension * sizeof(float));

        // pinned -> device
        cudaMemcpyAsync(d_batches[i], h_batch_pinned, batch_size * dimension * sizeof(float), cudaMemcpyHostToDevice, stream[i]);

        launch_cosine_similarity(d_batches[i], d_input, d_outputs[i], batch_size, dimension, TPB, stream[i]);

        // device -> pinned
        cudaMemcpyAsync(h_out_pinned, d_outputs[i], batch_size * sizeof(float),  cudaMemcpyDeviceToHost, stream[i]);

        // pinned -> pageable
        cudaStreamSynchronize(stream[i]);
        std::memcpy(h_output.data() + start_index, h_out_pinned, batch_size * sizeof(float));
    }

    for(int i = 0; i < num_mini_batches; i++){
        cudaFree(d_batches[i]);
        cudaFree(d_outputs[i]);
        // cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_input);
    cudaFreeHost(h_batch_pinned);
    cudaFreeHost(h_out_pinned);

    auto total_kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_kernel_elapsed = total_kernel_end - total_kernel_start;
    std::cout << "pinned_page Kernel time: " << total_kernel_elapsed.count() << " ms" << std::endl;

    std::cout << "[0] Cosine similarity result vs all:\n";
    for(int i = 0; i < std::min(N, (size_t)10); i++){
        std::cout << "sim[0][" << i << "] = " << h_output[i] << "\n";
    }

#if FileFlag
    std::ofstream File("cosine_result.txt");
    // Write final results to File
    for(int i = 0; i < N; i++){
        File << "sim[0][" << i << "] = " << h_output[i] << "\n";
    }
    File.close();
#endif

    return 0;
}
