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
    const int num_pools = 16;

    std::vector<cudaStream_t> stream(num_streams);

    std::vector<cudaEvent_t> events(num_pools);
    std::vector<float*> h_in_pinned(num_pools);
    std::vector<float*> h_out_pinned(num_pools);
    std::vector<float*> d_batches(num_pools);
    std::vector<float*> d_outputs(num_pools);

    std::vector<float> h_samples(N * dimension);
    std::vector<float> h_output(N);
    std::cout << "=== generate_sample_data ===\n";
    generate_sample_data(h_samples.data(), N, dimension);
    std::cout << "=== pinned code start ===" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < num_streams; i++){
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    for(int i = 0; i < num_pools; i++){
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
    int input_slot = 0; // buffer slot index
    for(int i = 0; i < num_batches; ++i) {
        std::size_t start_idx = i * fixed_batch_size;
        std::size_t end_idx = std::min(start_idx + fixed_batch_size, N);
        std::size_t batch_size = end_idx - start_idx;

        int stream_idx = i % num_streams;
        int slot = input_slot % num_pools;

        if(i >= num_pools){
            if(cudaEventQuery(events[slot]) != cudaSuccess){
                CUDA_CHECK(cudaEventSynchronize(events[slot]));
            }
            std::memcpy(h_output.data() + slot * fixed_batch_size, h_out_pinned[slot], fixed_batch_size * sizeof(float));
        }

        std::memcpy(h_in_pinned[slot], h_samples.data() + start_idx * dimension, batch_size * dimension * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(d_batches[slot], h_in_pinned[slot], batch_size * dimension * sizeof(float), cudaMemcpyHostToDevice, stream[stream_idx]));

        launch_cosine_similarity(d_batches[slot], d_input, d_outputs[slot], batch_size, dimension, TPB, stream[stream_idx]);

        CUDA_CHECK(cudaMemcpyAsync(h_out_pinned[slot], d_outputs[slot], batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream[stream_idx]));
        CUDA_CHECK(cudaEventRecord(events[slot], stream[stream_idx]));

        input_slot++;
    }

    // flush remaining stream
    for (int j = std::max(0, input_slot - num_pools); j < input_slot; ++j) {
        int slot = j % num_pools;
        CUDA_CHECK(cudaEventSynchronize(events[slot]));
        std::size_t start_idx = j * fixed_batch_size;
        std::size_t end_idx = std::min(start_idx + fixed_batch_size, N);
        std::size_t batch_size = end_idx - start_idx;
        std::memcpy(h_output.data() + start_idx, h_out_pinned[slot], batch_size * sizeof(float));
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
