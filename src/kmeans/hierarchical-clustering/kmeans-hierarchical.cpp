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

    if (argc != 6 && argc != 7) {
        std::cerr << "Usage for standard k-means: " << argv[0] << " N TPB K MAX_ITER DIM" << std::endl;
        std::cerr << "Usage for hierarchical clustering: " << argv[0] << " N TPB k_coarse k_fine MAX_ITER DIM" << std::endl;
        return 1;
    }

    std::size_t N = std::atoi(argv[1]); // Number of data points
    int TPB = std::atoi(argv[2]); // Threads per block
    // std::size_t K = std::atoi(argv[3]); // Number of clusters
    int MAX_ITER = std::atoi(argv[argc - 2]);
    std::size_t DIM = std::atoi(argv[argc - 1]); // Dimension of data points

    float *d_samples = nullptr, *d_clusterCenters = nullptr;
    int *d_clusterIndices = nullptr, *d_clusterSizes = nullptr;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device Setting---------------------------------" << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GBs" << std::endl;
    std::cout << "Size of Warp: " << prop.warpSize << std::endl;
    std::cout << "Max Threads Per Block(TPB): " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "chunk size: " << 256 << std::endl;
    std::cout << "TPB : " << TPB << std::endl;
    // std::cout << "Get Shared Memory : " << K * 256 * sizeof(float) << " bytes" <<  std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    if (argc == 6) {
        // === Standard k-means clustering ===
        std::size_t K = std::atoi(argv[3]); // number of cluster

        // GPU memory assignment
        float *d_samples = nullptr, *d_clusterCenters = nullptr;
        int *d_clusterIndices = nullptr, *d_clusterSizes = nullptr;
        cudaMalloc(&d_samples, N * DIM * sizeof(float));
        cudaMalloc(&d_clusterIndices, N * sizeof(int));
        cudaMalloc(&d_clusterCenters, K * DIM * sizeof(float));
        cudaMalloc(&d_clusterSizes, K * sizeof(int));
        cudaMemset(d_clusterIndices, -1, N * sizeof(int));
        cudaMemset(d_clusterSizes, 0, K * sizeof(int));

        std::vector<float> h_clusterCenters(K * DIM), h_samples(N * DIM);
        int *h_clusterIndices = (int*)malloc(N * sizeof(int));

        generate_sample_data(h_samples, h_clusterCenters, N, K, DIM);
        cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_samples, h_samples.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();

        for (int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter) {
            launch_kmeans_labeling(d_samples, d_clusterIndices, d_clusterCenters, N, TPB, K, DIM);
            cudaDeviceSynchronize();
            launch_kmeans_update_center(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, TPB, K, DIM);
            cudaDeviceSynchronize();

            cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * DIM * sizeof(float), cudaMemcpyDeviceToHost);
            double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, DIM);
            std::cout << "Iteration " << cur_iter << ": SSE = " << sse << std::endl;
        }
        cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "K-means execution time: " << elapsed.count() << " ms" << std::endl;

#if FileFlag
        std::ofstream File("kmeans_result.txt");
        File << "Final Centroids:\n";
        for (std::size_t k = 0; k < K; ++k) {
            File << "Centroid " << k << ": ";
            for (std::size_t d = 0; d < DIM; ++d) {
                File << h_clusterCenters[k * DIM + d] << " ";
            }
            File << "\n";
        }
        File << "\nData Points:\n";
        for (std::size_t i = 0; i < N; ++i) {
            File << "Data Point " << i << ": ";
            for (std::size_t d = 0; d < DIM; ++d) {
                File << h_samples[i * DIM + d] << " ";
            }
            File << " -> Cluster " << h_clusterIndices[i] << "\n";
        }
        File.close();
#endif

        cudaFree(d_samples);
        cudaFree(d_clusterIndices);
        cudaFree(d_clusterCenters);
        cudaFree(d_clusterSizes);
        free(h_clusterIndices);
    }
    else {
        // === Hierarchical clustering ===
        // arg: N, TPB, k_coarse, k_fine, MAX_ITER, DIM
        int k_coarse = std::atoi(argv[3]); // number of coarse cluster
        int k_fine = std::atoi(argv[4]);   // number of fine cluster

        // Coarse clustering
        float *d_samples = nullptr, *d_coarseCentroids = nullptr;
        int *d_coarseIndices = nullptr, *d_coarseSizes = nullptr;
        cudaMalloc(&d_samples, N * DIM * sizeof(float));
        cudaMalloc(&d_coarseIndices, N * sizeof(int));
        cudaMalloc(&d_coarseCentroids, k_coarse * DIM * sizeof(float));
        cudaMalloc(&d_coarseSizes, k_coarse * sizeof(int));
        cudaMemset(d_coarseIndices, -1, N * sizeof(int));
        cudaMemset(d_coarseSizes, 0, k_coarse * sizeof(int));

        
        std::vector<float> h_coarseCenters(k_coarse * DIM), h_samples;
        generate_sample_data(h_samples, h_coarseCenters, N, k_coarse*k_fine, DIM);
        cudaMemcpy(d_coarseCentroids, h_coarseCenters.data(), k_coarse * DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_samples, h_samples.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice);

        auto coarse_start = std::chrono::high_resolution_clock::now();
        std::vector<int> h_coarseIndices(N);
        std::vector<float> h_coarseCenters_out(k_coarse * DIM);
        for (int cur_iter = 1; cur_iter <= MAX_ITER/2; ++cur_iter) {
            launch_kmeans_labeling(d_samples, d_coarseIndices, d_coarseCentroids, N, TPB, k_coarse, DIM);
            cudaDeviceSynchronize();
            launch_kmeans_update_center(d_samples, d_coarseIndices, d_coarseCentroids, d_coarseSizes, N, TPB, k_coarse, DIM);
            cudaDeviceSynchronize();

            cudaMemcpy(h_coarseIndices.data(), d_coarseIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_coarseCenters_out.data(), d_coarseCentroids, k_coarse * DIM * sizeof(float), cudaMemcpyDeviceToHost);
            double coarse_sse = compute_SSE(h_samples, h_coarseCenters_out, h_coarseIndices, N, k_coarse, DIM);
            std::cout << "Coarse Iteration " << cur_iter << ": SSE = " << coarse_sse << std::endl;
        }
        auto coarse_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> coarse_elapsed = coarse_end - coarse_start;
        std::cout << "Coarse clustering execution time: " << coarse_elapsed.count() << " ms" << std::endl;


        std::vector<float> h_allSamples = h_samples;
        std::vector<int> h_fineIndices(N, -1);
        std::vector< std::vector<float> > final_fine_centers(k_coarse, std::vector<float>(k_fine * DIM, 0));

        auto fine_total_start = std::chrono::high_resolution_clock::now();

        for (int c = 0; c < k_coarse; c++) {
            std::vector<int> indices;
            for (int i = 0; i < N; i++) {
                if (h_coarseIndices[i] == c)
                    indices.push_back(i);
            }
            int sub_N = indices.size();
            if (sub_N == 0) continue; 

            std::vector<float> h_subData(sub_N * DIM);
            for (int i = 0; i < sub_N; i++) {
                int idx = indices[i];
                for (int d = 0; d < DIM; d++) {
                    h_subData[i * DIM + d] = h_allSamples[idx * DIM + d];
                }
            }

            std::vector<float> h_subCentroids(k_fine * DIM);
            for (int j = 0; j < k_fine; j++) {
                int idx = j % sub_N;
                for (int d = 0; d < DIM; d++) {
                    h_subCentroids[j * DIM + d] = h_subData[idx * DIM + d];
                }
            }

            float *d_subData = nullptr, *d_subCentroids = nullptr;
            int *d_subIndices = nullptr, *d_subSizes = nullptr;
            cudaMalloc(&d_subData, sub_N * DIM * sizeof(float));
            cudaMalloc(&d_subCentroids, k_fine * DIM * sizeof(float));
            cudaMalloc(&d_subIndices, sub_N * sizeof(int));
            cudaMalloc(&d_subSizes, k_fine * sizeof(int));
            cudaMemcpy(d_subData, h_subData.data(), sub_N * DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_subCentroids, h_subCentroids.data(), k_fine * DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_subIndices, -1, sub_N * sizeof(int));
            cudaMemset(d_subSizes, 0, k_fine * sizeof(int));

            auto fine_start = std::chrono::high_resolution_clock::now();
            std::vector<int> h_subIndices(sub_N);
            std::vector<float> h_subCentroids_out(k_fine * DIM);
            for (int cur_iter = 1; cur_iter <= MAX_ITER/2; ++cur_iter) {
                launch_kmeans_labeling(d_subData, d_subIndices, d_subCentroids, sub_N, TPB, k_fine, DIM);
                cudaDeviceSynchronize();
                launch_kmeans_update_center(d_subData, d_subIndices, d_subCentroids, d_subSizes, sub_N, TPB, k_fine, DIM);
                cudaDeviceSynchronize();

                cudaMemcpy(h_subIndices.data(), d_subIndices, sub_N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_subCentroids_out.data(), d_subCentroids, k_fine * DIM * sizeof(float), cudaMemcpyDeviceToHost);
                double fine_sse = compute_SSE(h_subData, h_subCentroids_out, h_subIndices, sub_N, k_fine, DIM);
                std::cout << "Coarse Cluster " << c << " - Fine Iteration " << cur_iter << ": SSE = " << fine_sse << std::endl;
            }
            auto fine_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> fine_elapsed = fine_end - fine_start;
            std::cout << "Coarse Cluster " << c << " - Fine clustering execution time: " << fine_elapsed.count() << " ms" << std::endl;

            final_fine_centers[c] = h_subCentroids_out; 

            cudaMemcpy(h_subIndices.data(), d_subIndices, sub_N * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < sub_N; i++) {
                int global_idx = indices[i];
                h_fineIndices[global_idx] = h_subIndices[i];
            }

            cudaFree(d_subData);
            cudaFree(d_subCentroids);
            cudaFree(d_subIndices);
            cudaFree(d_subSizes);
        }
        auto fine_total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fine_total_elapsed = fine_total_end - fine_total_start;
        std::cout << "Total Fine clustering execution time: " << fine_total_elapsed.count() << " ms" << std::endl;


        double hierarchical_global_sse = 0.0;
        for (int i = 0; i < N; i++) {
            int c = h_coarseIndices[i]; 
            int f = h_fineIndices[i];    
            if (c < 0 || f < 0) continue; 

            double sqDist = 0.0;
            for (int d = 0; d < DIM; d++) {
                float diff = h_samples[i * DIM + d] - final_fine_centers[c][f * DIM + d];
                sqDist += diff * diff;
            }
            hierarchical_global_sse += sqDist;
        }
        std::cout << "Hierarchical Global SSE: " << hierarchical_global_sse << std::endl;


        // Free coarse clustering GPU memory
        cudaFree(d_samples);
        cudaFree(d_coarseIndices);
        cudaFree(d_coarseCentroids);
        cudaFree(d_coarseSizes);
    }
    
    return 0;
}