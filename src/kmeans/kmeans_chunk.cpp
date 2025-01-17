#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <random>
#include <limits>
#include "kmeans.h"

#define FileFlag 0

// Function to generate clustered data
template<typename VecType = float>
void generate_sample_data(std::vector<VecType>& h_data, std::vector<VecType>& h_clusterCenters, std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = std::numeric_limits<std::size_t>::max()) {
    std::random_device random_device;
    std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? random_device() : static_cast<unsigned int>(seed));

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)0.001);
    std::uniform_int_distribution<std::size_t> idxUnit(0, K - 1);
    std::normal_distribution<VecType> norm((VecType)0, (VecType)0.025);

    h_data.resize(N * DIM);
    h_clusterCenters.resize(K * DIM);

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t dim = 0; dim < DIM; ++dim) {
            h_clusterCenters[k * DIM + dim] = vecUnit(generator);
        }
    }

    for (std::size_t n = 0; n < N; ++n) {
        std::size_t cur_index = idxUnit(generator);
        for (std::size_t dim = 0; dim < DIM; ++dim) {
            h_data[n * DIM + dim] = h_clusterCenters[cur_index * dim + dim] + norm(generator);
        }
    }
    return;
}

double compute_SSE(const std::vector<float>& data, const std::vector<float>& centroids,
                  const std::vector<int>& clusterIndices, std::size_t N, std::size_t K, std::size_t DIM) {
    double sse = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
        double squaredDistance = 0.0;
        for (std::size_t d = 0; d < DIM; ++d) {
            float vecDiff  = data[n * DIM + d] - centroids[clusterIndices[n] * DIM + d];
            squaredDistance += vecDiff  * vecDiff ;
        }
        sse += squaredDistance;
    }
    return sse;
}

int main(int argc, char *argv[])
{
    std::cout.precision(10);

    if (argc != 5) {
        return 1;
    }

    std::size_t N = std::atoi(argv[1]); // Number of data points
    std::size_t K = std::atoi(argv[2]); // Number of clusters
    int MAX_ITER = std::atoi(argv[3]);
    std::size_t dimension = std::atoi(argv[4]); // Dimension of data points

    float *d_samples = nullptr, *d_clusterCenters = nullptr;
    int *d_clusterIndices = nullptr, *d_clusterSizes = nullptr;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t maxChunkSize = prop.sharedMemPerBlock / (K * sizeof(float));
    int chunkSize = (int)maxChunkSize - 1;
    // chunkSize = (chunkSize / prop.warpSize) * prop.warpSize;
    int TPB = std::__bit_floor(chunkSize) > prop.maxThreadsPerBlock / 2 ? prop.maxThreadsPerBlock / 2 : std::__bit_floor(chunkSize);

    std::cout << "Device Setting---------------------------------" << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GBs" << std::endl;
    std::cout << "Size of Warp: " << prop.warpSize << std::endl;
    std::cout << "Max Threads Per Block(TPB): " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "chunk size: " << chunkSize << std::endl;
    std::cout << "TPB : " << TPB << std::endl;
    std::cout << "Get Shared Memory : " << K * chunkSize * sizeof(float) << " bytes" <<  std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    // Allocate GPU memory
    auto mallocStart = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMalloc(&d_clusterIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCenters, K * dimension * sizeof(float));
    cudaMalloc(&d_clusterSizes, K * sizeof(int));
    auto mallocEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mallocElapsed = mallocEnd - mallocStart;
    std::cout << "Malloc time: " << mallocElapsed.count() << " ms" << std::endl;

    // Initialize GPU memory
    auto memsetStart = std::chrono::high_resolution_clock::now();
    cudaMemset(d_clusterIndices, -1, N * sizeof(int));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));
    auto memsetEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memsetElapsed = memsetEnd - memsetStart;
    std::cout << "Memset time: " << memsetElapsed.count() << " ms" << std::endl;

    // Generate data
    auto generateStart = std::chrono::high_resolution_clock::now();
    std::vector<float> h_clusterCenters(K * dimension), h_samples(N * dimension);
    int *h_clusterIndices = (int*)malloc(N * sizeof(int));
    generate_sample_data(h_samples, h_clusterCenters, N, K, dimension);
    auto generateEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> generateElapsed = generateEnd - generateStart;
    std::cout << "Generate data time: " << generateElapsed.count() << " ms" << std::endl;

    // Copy data to GPU
    auto copyStart = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
    auto copyEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copyElapsed = copyEnd - copyStart;
    std::cout << "Copy data time: " << copyElapsed.count() << " ms" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for(int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter)
    {
        // Cluster assignment step
        launch_kmeans_labeling_chunk(d_samples, d_clusterIndices, d_clusterCenters, N, TPB, K, dimension, chunkSize);
        cudaDeviceSynchronize();

        // Centroid update step
        launch_kmeans_update_center_chunk(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, TPB, K, dimension, chunkSize);
        cudaDeviceSynchronize();

        cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);
        if(cur_iter % 10 == 0 || cur_iter == 1)
        {
            double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
            std::cout << "Iteration " << cur_iter << " SSE = " << sse << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);
    double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
    std::cout << "SSE = " << sse << std::endl;

    std::cout << "K-means execution time: " << elapsed.count() << " ms" << std::endl;

#if FileFlag
    std::ofstream File("kmeans_result.txt");
    // Write final results to File
    File << "Final Centroids : \n";
    for (std::size_t k = 0; k < K; ++k) {
        File << "Centroid " << k << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_clusterCenters[k * dimension + d] << " ";
        }
        File << "\n";
    }

    // File << "\nData Points:\n";
    // for (std::size_t i = 0; i < N; ++i) {
    //     File << "Data Point " << i << ": ";
    //     for (std::size_t d = 0; d < dimension; ++d) {
    //         File << h_samples[i * dimension + d] << " ";
    //     }
    //     File << " -> Cluster " << h_clusterIndices[i] << "\n";
    // }

    File.close();
#endif

    cudaFree(d_samples);
    cudaFree(d_clusterIndices);
    cudaFree(d_clusterCenters);
    cudaFree(d_clusterSizes);

    free(h_clusterIndices);

    return 0;
}