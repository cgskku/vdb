#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "cosine.h"

#define FileFlag 1

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
    if(code != cudaSuccess){
        std::cerr << "GPUassert: " << cudaGetErrorString(code)
                  << " " << file << ":" << line << std::endl;
        std::exit(code);
    }
}

int main(int argc, char *argv[])
{
    std::cout.precision(10);
    std::size_t N = 1000000;
    std::size_t dimension = 1536;

    int TPB = 128;
    int fixed_batch_size = 10000;
    int num_batches = (N + fixed_batch_size - 1) / fixed_batch_size;

    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDevice(&cuda_device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, cuda_device));
    std::cout << "Device: " << deviceProp.name << "\n";
    std::cout << "asyncEngineCount: " << deviceProp.asyncEngineCount << "\n";

    const int num_streams = deviceProp.asyncEngineCount;
    std::vector<cudaStream_t> stream(num_streams);
    std::vector<cudaEvent_t> events(num_streams);
    std::vector<float*> h_in_pinned(num_streams);
    std::vector<float*> h_out_pinned(num_streams);
    std::vector<float*> d_batches(num_streams);
    std::vector<float*> d_outputs(num_streams);

    std::vector<float> h_samples(N * dimension);
    std::vector<float> h_output(N);
    std::cout << "=== generate_sample_data ===\n";
    generate_sample_data(h_samples.data(), N, dimension);
    std::cout << "=== pinned code start ===" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < num_streams; i++){
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&events[i]));
        CUDA_CHECK(cudaHostAlloc(&h_in_pinned[i], fixed_batch_size * dimension * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_out_pinned[i], fixed_batch_size * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&d_batches[i], fixed_batch_size * dimension * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputs[i], fixed_batch_size * sizeof(float)));
    }

    float* d_input = nullptr;
    cudaMalloc(&d_input, dimension * sizeof(float));
    cudaMemcpyAsync(d_input, &h_samples[0], dimension * sizeof(float), cudaMemcpyHostToDevice, stream[0]);

    auto total_for_start = std::chrono::high_resolution_clock::now();
    // num_batches = 2
    for(int i = 0; i < num_batches; i++){
        int cur = i % num_streams;

        std::size_t start_idx = i * fixed_batch_size;
        std::size_t end_idx = std::min(start_idx + fixed_batch_size, N);
        std::size_t batch_size = end_idx - start_idx;

        // CUDA_CHECK(cudaEventSynchronize(events[cur]));

        std::memcpy(h_in_pinned[cur], h_samples.data() + start_idx * dimension, batch_size * dimension * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(d_batches[cur], h_in_pinned[cur],
            batch_size * dimension * sizeof(float), cudaMemcpyHostToDevice, stream[cur]));

        launch_cosine_similarity(d_batches[cur], d_input, d_outputs[cur],
            batch_size, dimension, TPB, stream[cur]);

        CUDA_CHECK(cudaMemcpyAsync(h_out_pinned[cur], d_outputs[cur],
            batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream[cur]));

        CUDA_CHECK(cudaEventRecord(events[cur], stream[cur]));
        CUDA_CHECK(cudaEventSynchronize(events[cur]));
        std::memcpy(h_output.data() + start_idx, h_out_pinned[cur], batch_size * sizeof(float));
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = total_end - total_start;
    std::chrono::duration<double, std::milli> for_elapsed = total_end - total_for_start;
    std::cout << "pinned_page loop time: " << for_elapsed.count() << " ms\n";
    std::cout << "pinned_page code time: " << elapsed.count() << " ms\n";

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

    for(int i = 0; i < num_streams; i++){
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(events[i]);
        cudaFree(d_batches[i]);
        cudaFree(d_outputs[i]);
        cudaFreeHost(h_in_pinned[i]);
        cudaFreeHost(h_out_pinned[i]);
    }
    cudaFree(d_input);

    return 0;
}
