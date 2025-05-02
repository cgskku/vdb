#include <iostream>
#include <fstream>
#include <cstdlib>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <vector>
#include <random>
#include <limits>

#include "cosine.h"

// Function to generate clustered data
template<typename VecType = float>
void generate_sample_data(
    std::vector<VecType>& h_data,
    std::size_t N, std::size_t DIM,
    std::size_t seed = std::numeric_limits<std::size_t>::max()) 
{
    std::random_device random_device;
    std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? random_device() : static_cast<unsigned int>(seed));

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)0.001);
    std::normal_distribution<VecType> norm((VecType)0, (VecType)0.025);

    h_data.resize(N * DIM);

    for(std::size_t n = 0; n < N; ++n){
        for(std::size_t dim = 0; dim < DIM; ++dim){
            h_data[n * DIM + dim] = vecUnit(generator);
        }
    }

    return;
}

// this program is used to test the cosine similarity kernel

int main(int argc, char *argv[])
{
    std::cout.precision(10);

    if(argc != 3){
        std::cerr << "Usage: " << argv[0] << " <N> <dimension>" << std::endl;
        return 1;
    }

    std::size_t N = std::atoi(argv[1]); // Number of data points
    std::size_t dimension = std::atoi(argv[2]); // Dimension of data points

    int TPB = 128;
    float *d_samples = nullptr;

    // Generate data
    std::vector<float> h_samples(N * dimension);
    generate_sample_data(h_samples, N, dimension);

    // Allocate GPU memory
    cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);

    float* d_input;
    cudaMalloc(&d_input, dimension * sizeof(float));
    cudaMemcpy(d_input, h_samples.data(), dimension * sizeof(float), cudaMemcpyHostToDevice);

    float* d_output;
    std::vector<float> h_output(N);
    cudaMalloc(&d_output, N * sizeof(float));

    auto kernelStart = std::chrono::high_resolution_clock::now();

    launch_cosine_similarity(d_samples, d_input, d_output, N, dimension, TPB);
    cudaDeviceSynchronize();

    auto kernelEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernelElapsed = kernelEnd - kernelStart;
    std::cout << "Kernel time: " << kernelElapsed.count() << " ms" << std::endl;

    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[0] Cosine similarity result vs all:\n";
    for(int i = 0; i < std::min(N, 10); i++)
        std::cout << "sim[0][" << i << "] = " << h_output[i] << "\n";

    // clean up
    cudaFree(d_samples);
    cudaFree(d_input);
    cudaFree(d_output);
}