#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include "kmeans.h"
#include "lsh.h" 

void generate_sample_data(std::vector<float>& h_data, std::vector<float>& h_centroids, std::size_t N, std::size_t K, std::size_t DIM) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> vecUnit(0.0f, 0.001f);
    std::uniform_int_distribution<int> idxDist(0, K - 1);
    std::normal_distribution<float> noise(0.0f, 0.025f);

    h_data.resize(N * DIM);
    h_centroids.resize(K * DIM);

    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t d = 0; d < DIM; ++d)
            h_centroids[k * DIM + d] = vecUnit(gen);

    for (std::size_t n = 0; n < N; ++n) {
        int cluster = idxDist(gen);
        for (std::size_t d = 0; d < DIM; ++d)
            h_data[n * DIM + d] = h_centroids[cluster * DIM + d] + noise(gen);
    }
}

void compute_group_centroids(const std::vector<float>& data, const std::vector<int>& group_ids, int num_groups, int DIM, std::vector<float>& group_centroids) {
    std::vector<int> group_sizes(num_groups, 0);
    group_centroids.assign(num_groups * DIM, 0.0f);

    for (std::size_t i = 0; i < group_ids.size(); ++i) {
        int g = group_ids[i];
        for (int d = 0; d < DIM; ++d)
            group_centroids[g * DIM + d] += data[i * DIM + d];
        group_sizes[g]++;
    }

    for (int g = 0; g < num_groups; ++g) {
        if (group_sizes[g] > 0) {
            for (int d = 0; d < DIM; ++d)
                group_centroids[g * DIM + d] /= group_sizes[g];
        }
    }
}

int main() {
    const int N = 1000, DIM = 1536, K_coarse = 25, K_group = 3, TPB = 256;
    const bool USE_LSH = true; 

    std::vector<float> h_data, h_centroids;
    generate_sample_data(h_data, h_centroids, N, K_coarse * 3, DIM);

    auto total_start = std::chrono::high_resolution_clock::now();

    float *d_data;
    int *d_labels;
    cudaMalloc(&d_data, N * DIM * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice);

    if (USE_LSH) {
        // === LSH-based coarse clustering ===
        int *d_lsh_labels;
        float *d_random_proj;
        cudaMalloc(&d_lsh_labels, N * sizeof(int));
        cudaMalloc(&d_random_proj, DIM * sizeof(float));

        std::vector<float> h_proj(DIM);
        std::mt19937 gen(123);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < DIM; ++i) h_proj[i] = dist(gen);
        cudaMemcpy(d_random_proj, h_proj.data(), DIM * sizeof(float), cudaMemcpyHostToDevice);

        launch_lsh(d_data, d_lsh_labels, d_random_proj, N, DIM, K_coarse);
        std::cout << "LSH 완료 " << std::endl;

        std::vector<int> h_coarseLabels(N);
        cudaMemcpy(h_coarseLabels.data(), d_lsh_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_lsh_labels);
        cudaFree(d_random_proj);

        std::vector<float> coarse_group_centroids;
        compute_group_centroids(h_data, h_coarseLabels, K_coarse, DIM, coarse_group_centroids);


    } else {
        // === KMeans-based coarse clustering ===
        float *d_centroids;
        int *d_sizes;
        cudaMalloc(&d_centroids, K_coarse * DIM * sizeof(float));
        cudaMalloc(&d_labels, N * sizeof(int));
        cudaMalloc(&d_sizes, K_coarse * sizeof(int));
        cudaMemcpy(d_centroids, h_centroids.data(), K_coarse * DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_labels, -1, N * sizeof(int));
        cudaMemset(d_sizes, 0, K_coarse * sizeof(int));

        std::vector<float> prev(K_coarse * DIM), cur(K_coarse * DIM);
        const double tol = 1e-5;
        int stable = 0;
        for (int i = 0; i < 10; ++i) {
            launch_kmeans_labeling(d_data, d_labels, d_centroids, N, TPB, K_coarse, DIM);
            launch_kmeans_update_center(d_data, d_labels, d_centroids, d_sizes, N, TPB, K_coarse, DIM);
            cudaMemcpy(cur.data(), d_centroids, K_coarse * DIM * sizeof(float), cudaMemcpyDeviceToHost);
            if (i > 0) {
                double diff = 0.0;
                for (int j = 0; j < K_coarse * DIM; ++j) diff += std::abs(cur[j] - prev[j]);
                if (diff < tol && ++stable >= 3){
                    std::cout << "Coarse clustering converged at iteration " << i << std::endl;
                    break;
                }
                if (diff >= tol) stable = 0;
            }
            prev = cur;
        }

        std::vector<int> h_coarseLabels(N);
        cudaMemcpy(h_coarseLabels.data(), d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<float> coarse_group_centroids;
        compute_group_centroids(h_data, h_coarseLabels, K_coarse, DIM, coarse_group_centroids);

        // === Final clustering on coarse group centroids ===
        float *d_group_centroids;
        int *d_group_labels, *d_group_sizes;
        cudaMalloc(&d_group_centroids, K_coarse * DIM * sizeof(float));
        cudaMalloc(&d_group_labels, K_coarse * sizeof(int));
        cudaMalloc(&d_group_sizes, K_group * sizeof(int));
        cudaMemcpy(d_group_centroids, coarse_group_centroids.data(), K_coarse * DIM * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < 10; ++i) {
            launch_kmeans_labeling(d_group_centroids, d_group_labels, d_group_centroids, K_coarse, TPB, K_group, DIM);
            launch_kmeans_update_center(d_group_centroids, d_group_labels, d_group_centroids, d_group_sizes, K_coarse, TPB, K_group, DIM);
        }

        std::vector<int> group_labels(K_coarse);
        cudaMemcpy(group_labels.data(), d_group_labels, K_coarse * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; ++i)
            h_coarseLabels[i] = group_labels[h_coarseLabels[i]];

        //Optional print
        // for (int i = 0; i < std::min(N, 50); ++i)
        //     std::cout << "Data Point " << i << " -> Final Cluster " << h_coarseLabels[i] << std::endl;

        cudaFree(d_centroids);
        cudaFree(d_sizes);
        cudaFree(d_group_centroids);
        cudaFree(d_group_labels);
        cudaFree(d_group_sizes);
    }

    cudaFree(d_data);
    cudaFree(d_labels);

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = total_end - total_start;
    std::cout << "Total execution time: " << elapsed.count() << " ms" << std::endl;

    return 0;
}
