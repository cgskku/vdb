#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <random>
#include <limits>
#include "kmeans-dc-accel.h"
#include <algorithm>

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

void generate_sample_data_with_labels(
    std::vector<float>& h_data, std::vector<float>& h_clusterCenters,
    std::vector<int>& gt_labels, std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = 42)
{
    std::mt19937 generator(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<float> vecUnit(-1.f, 1.0005f);
    std::uniform_int_distribution<std::size_t> idxUnit(0, K - 1);
    std::normal_distribution<float> norm(0.f, 0.001f);

    h_data.resize(N * DIM);
    h_clusterCenters.resize(K * DIM);
    gt_labels.resize(N);

    // Random cluster centers
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t d = 0; d < DIM; ++d)
            h_clusterCenters[k * DIM + d] = vecUnit(generator);

    // Generate data points and record true label
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t cid = idxUnit(generator);
        gt_labels[i] = static_cast<int>(cid);
        for (std::size_t d = 0; d < DIM; ++d)
            h_data[i * DIM + d] = h_clusterCenters[cid * DIM + d] + norm(generator);
    }
}

void pick_initial_centroids_shuffle(
    const std::vector<float>& data,  // [N * DIM]
    std::vector<float>& centroids,   // [K * DIM] (output)
    std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = 12345)
{
    std::vector<std::size_t> idxs(N);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idxs.begin(), idxs.end(), rng);
    for (std::size_t k = 0; k < K; ++k) {
        std::size_t sel = idxs[k];
        for (std::size_t d = 0; d < DIM; ++d)
            centroids[k * DIM + d] = data[sel * DIM + d];
    }
}

void pick_initial_centroids_by_gt(
    const std::vector<float>& data,     
    const std::vector<int>& gt_labels,  
    std::vector<float>& centroids,     
    std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = 12345)
{
    std::mt19937 rng(seed);
    for (std::size_t k = 0; k < K; ++k) {
        std::vector<std::size_t> idxs_for_label;
        for (std::size_t i = 0; i < N; ++i)
            if (gt_labels[i] == static_cast<int>(k))
                idxs_for_label.push_back(i);
        if (idxs_for_label.empty()) {
            std::cerr << "Warning: no samples for GT label " << k << std::endl;
            for (std::size_t d = 0; d < DIM; ++d)
                centroids[k * DIM + d] = 0; 
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, idxs_for_label.size() - 1);
            std::size_t sel = idxs_for_label[pick(rng)];
            for (std::size_t d = 0; d < DIM; ++d)
                centroids[k * DIM + d] = data[sel * DIM + d];
        }
    }
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

    if (argc != 7) {
        return 1;
    }

    std::size_t N = std::atoi(argv[1]); // Number of data points
    int TPB = std::atoi(argv[2]); // Threads per block
    std::size_t K = std::atoi(argv[3]); // Number of clusters
    int MAX_ITER = std::atoi(argv[4]);
    std::size_t dimension = std::atoi(argv[5]); // Dimension of data points
    std::string norm_str = argv[6];
    NormType normType;
    if (norm_str == "l1" || norm_str == "L1") normType = L1_NORM;
    else if (norm_str == "l2" || norm_str == "L2") normType = L2_NORM;
    else {
        std::cerr << "Invalid norm type. Use 'l1' or 'l2'.\n";
        return 1;
    }

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
    std::cout << "Get Shared Memory : " << K * 256 * sizeof(float) << " bytes" <<  std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    // Allocate GPU memory
    cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMalloc(&d_clusterIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCenters, K * dimension * sizeof(float));
    cudaMalloc(&d_clusterSizes, K * sizeof(int));

    cudaMemset(d_clusterIndices, -1, N * sizeof(int));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));

    std::vector<float> h_clusterCenters(K * dimension), h_samples(N * dimension);
    int *h_clusterIndices = (int*)malloc(N * sizeof(int));

    std::vector<int> gt_labels;
    generate_sample_data_with_labels(h_samples, h_clusterCenters, gt_labels, N, K, dimension);


    std::vector<float> h_kmeansCenters(K * dimension);
    pick_initial_centroids_shuffle(h_samples, h_kmeansCenters, N, K, dimension, 42);
    //pick_initial_centroids_by_gt(h_samples, gt_labels, h_kmeansCenters, N, K, dimension, 42);


    std::vector<int> gt_label_counts(K, 0);
    for (std::size_t i = 0; i < N; ++i)
        ++gt_label_counts[gt_labels[i]];

    std::cout << "[Before clustering] Ground-truth label distribution:" << std::endl;
    for (std::size_t k = 0; k < K; ++k)
        std::cout << "  Label " << k << ": " << gt_label_counts[k] << std::endl;
    

    // cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusterCenters, h_kmeansCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);


    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> prev_centroids(K * dimension, 0.f);

    for(int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter)
    {
        std::copy(h_clusterCenters.begin(), h_clusterCenters.end(), prev_centroids.begin());

        // Cluster assignment step
        launch_kmeans_labeling(d_samples, d_clusterIndices, d_clusterCenters, N, TPB, K, dimension, normType);
        cudaDeviceSynchronize();

        // Centroid update step
        launch_kmeans_update_center(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, TPB, K, dimension);
        cudaDeviceSynchronize();

        cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);

        double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
        std::cout << "Iteration " << cur_iter << ": SSE = " << sse << std::endl;

        double diff = 0.0;
        for (std::size_t i = 0; i < K * dimension; ++i)
            diff += (h_clusterCenters[i] - prev_centroids[i]) * (h_clusterCenters[i] - prev_centroids[i]);
        diff = std::sqrt(diff);

        std::cout << "    Centroid move L2 = " << diff << std::endl;

        const double threshold = 1e-5; 
        if (diff < threshold) {
            std::cout << "Converged at iteration " << cur_iter << " (centroid movement < " << threshold << ")" << std::endl;
            break;
        }
    }
    cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "K-means execution time: " << elapsed.count() << " ms" << std::endl;

    std::vector<std::vector<int>> cluster_gtlabel_count(K, std::vector<int>(K, 0));
    for (std::size_t i = 0; i < N; ++i) {
        int assigned = h_clusterIndices[i]; 
        int gt = gt_labels[i];           
        if (assigned >= 0 && assigned < (int)K && gt >= 0 && gt < (int)K)
            ++cluster_gtlabel_count[assigned][gt];
    }

    std::cout << "\n[After clustering] For each cluster, count of each GT label:" << std::endl;
    for (std::size_t c = 0; c < K; ++c) {
        std::cout << "Cluster " << c << ": ";
        for (std::size_t g = 0; g < K; ++g) {
            std::cout << "GT" << g << ": " << cluster_gtlabel_count[c][g] << " ";
        }
        std::cout << std::endl;
    }

#if FileFlag
    std::ofstream File("kmeans_result.txt");
    // Write final results to File
    File << "Final Centroids:\n";
    for (std::size_t k = 0; k < K; ++k) {
        File << "Centroid " << k << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_clusterCenters[k * dimension + d] << " ";
        }
        File << "\n";
    }


    File << "\nData Points:\n";
    for (std::size_t i = 0; i < N; ++i) {
        File << "Data Point " << i << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_samples[i * dimension + d] << " ";
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

    return 0;
}