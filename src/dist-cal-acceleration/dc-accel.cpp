#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include "dc-accel.h"

#define FileFlag 0

void generate_clustered_vectors(
    std::vector<float>& db_vectors,
    std::vector<float>& cluster_centers,
    std::vector<int>& labels,
    std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = 42)
{
    std::mt19937 generator(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<float> center_dist(-1.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.02f);

    cluster_centers.resize(K * DIM);
    db_vectors.resize(N * DIM);
    labels.resize(N);

    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t d = 0; d < DIM; ++d)
            cluster_centers[k * DIM + d] = center_dist(generator);

    std::uniform_int_distribution<std::size_t> pick_cluster(0, K - 1);
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t cid = pick_cluster(generator);
        labels[i] = static_cast<int>(cid);
        for (std::size_t d = 0; d < DIM; ++d)
            db_vectors[i * DIM + d] = cluster_centers[cid * DIM + d] + noise_dist(generator);
    }
}

void generate_query_vector(
    std::vector<float>& query,
    const std::vector<float>& cluster_centers,
    std::size_t DIM,
    std::size_t seed = 123,
    int cluster_idx = -1) // -1: random, 0~K-1: near cluster
{
    std::mt19937 generator(static_cast<unsigned int>(seed));
    std::normal_distribution<float> noise_dist(0.0f, 0.02f);

    query.resize(DIM);

    if (cluster_idx >= 0 && !cluster_centers.empty()) {
        for (std::size_t d = 0; d < DIM; ++d)
            query[d] = cluster_centers[cluster_idx * DIM + d] + noise_dist(generator);
    } 
    else {
        std::uniform_real_distribution<float> rand_dist(-1.0f, 1.0f);
        for (std::size_t d = 0; d < DIM; ++d)
            query[d] = rand_dist(generator);
    }
}

float cpu_distance_l1(const float* a, const float* b, std::size_t DIM) {
    float dist = 0.0f;
    for (std::size_t d = 0; d < DIM; ++d)
        dist += std::abs(a[d] - b[d]);
    return dist;
}

float cpu_distance_l2(const float* a, const float* b, std::size_t DIM) {
    float dist = 0.0f;
    for (std::size_t d = 0; d < DIM; ++d) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

float cpu_distance_cosine(const float* a, const float* b, std::size_t DIM) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (std::size_t d = 0; d < DIM; ++d) {
        dot += a[d] * b[d];
        na += a[d] * a[d];
        nb += b[d] * b[d];
    }
    float denom = std::sqrt(na) * std::sqrt(nb) + 0.00000001f;
    float cosine_similarity = dot / denom;
    return 1.0f - cosine_similarity;
}

int main(int argc, char *argv[])
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " N TPB K DIM norm_type(l1|l2|cosine)\n";
        return 1;
    }

    std::size_t N = std::atoi(argv[1]);
    int TPB = std::atoi(argv[2]);
    std::size_t K = std::atoi(argv[3]);
    std::size_t DIM = std::atoi(argv[4]);
    std::string norm_str = argv[5];
    int QUERY_GT_LABEL = std::atoi(argv[6]);

    NormType normType;
    if (norm_str == "l1" || norm_str == "L1") normType = L1_NORM;
    else if (norm_str == "l2" || norm_str == "L2") normType = L2_NORM;
    else if (norm_str == "cosine" || norm_str == "COSINE") normType = COSINE_DIST;
    else {
        std::cerr << "Invalid norm type. Use 'l1', 'l2', or 'cosine'.\n";
        return 1;
    }

    std::vector<float> db_vectors, cluster_centers;
    std::vector<int> labels;
    generate_clustered_vectors(db_vectors, cluster_centers, labels, N, K, DIM, 42);

    std::vector<float> query;
    generate_query_vector(query, cluster_centers, DIM, 314, QUERY_GT_LABEL);

    float *d_db_vectors, *d_query, *d_dists;
    cudaMalloc(&d_db_vectors, N * DIM * sizeof(float));
    cudaMalloc(&d_query, DIM * sizeof(float));
    cudaMalloc(&d_dists, N * sizeof(float));
    cudaMemcpy(d_db_vectors, db_vectors.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query.data(), DIM * sizeof(float), cudaMemcpyHostToDevice);


    // gpu query-db
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    launch_distance_kernel(d_db_vectors, d_query, d_dists, N, DIM, normType, TPB);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Distance computation time: " << milliseconds << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<float> h_dists(N);
    cudaMemcpy(h_dists.data(), d_dists, N * sizeof(float), cudaMemcpyDeviceToHost);

    float min_dist = 1e10f;
    int min_idx = -1;
    for (std::size_t i = 0; i < N; ++i) {
        if (h_dists[i] < min_dist) {
            min_dist = h_dists[i];
            min_idx = static_cast<int>(i);
        }
    }
    std::cout << "[VectorDB] min_idx: " << min_idx << ", min_dist: " << min_dist << ", label(min_idx): " << labels[min_idx] << std::endl;

    int query_gt_label = QUERY_GT_LABEL;

    std::vector<std::pair<float, int>> dist_idx_pairs;
    for (std::size_t i = 0; i < N; ++i)
        dist_idx_pairs.emplace_back(h_dists[i], i);
    std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());

    int topk = 10;
    int correct_label_count = 0;
    std::cout << "Top-" << topk << " nearest vectors:\n";
    for (int i = 0; i < topk; ++i) {
        int idx = dist_idx_pairs[i].second;
        std::cout << "  idx: " << idx << ", dist: " << dist_idx_pairs[i].first << ", label: " << labels[idx] << "\n";
        if (labels[idx] == query_gt_label)
            correct_label_count++;
    }
    std::cout << "Correct label count in Top-" << topk << ": " << correct_label_count << std::endl;

    // gpu db-db
    float *d_pairwise;
    cudaMalloc(&d_pairwise, N * N * sizeof(float));

    int blockX = 16, blockY = 16;

    cudaEvent_t pw_start, pw_stop;
    cudaEventCreate(&pw_start); cudaEventCreate(&pw_stop);
    cudaEventRecord(pw_start, 0);

    launch_pairwise_distance_kernel(d_db_vectors, d_pairwise, N, DIM, normType, blockX, blockY);

    cudaEventRecord(pw_stop, 0);
    cudaEventSynchronize(pw_stop);

    float pw_msec = 0;
    cudaEventElapsedTime(&pw_msec, pw_start, pw_stop);
    std::cout << "[GPU] Pairwise (all-to-all) distance matrix computed in " << pw_msec << " ms" << std::endl;
    cudaEventDestroy(pw_start); cudaEventDestroy(pw_stop);

    std::vector<float> h_pairwise(std::min<size_t>(N*N, 100));
    cudaMemcpy(h_pairwise.data(), d_pairwise, h_pairwise.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[GPU] Pairwise distance sample (first 3x3):" << std::endl;
    for (int i = 0; i < std::min<int>(3, N); ++i) {
        for (int j = 0; j < std::min<int>(3, N); ++j)
            std::cout << h_pairwise[i*N + j] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_pairwise);

    cudaFree(d_db_vectors); cudaFree(d_query); cudaFree(d_dists);

    // cpu
    auto cpu_start = std::chrono::high_resolution_clock::now();

    std::vector<float> cpu_dists(N);
    float cpu_min_dist = 1e10f;
    int cpu_min_idx = -1;

    for (std::size_t i = 0; i < N; ++i) {
        float dist;
        if (normType == L1_NORM)
            dist = cpu_distance_l1(query.data(), db_vectors.data() + i * DIM, DIM);
        else if (normType == L2_NORM)
            dist = cpu_distance_l2(query.data(), db_vectors.data() + i * DIM, DIM);
        else
            dist = cpu_distance_cosine(query.data(), db_vectors.data() + i * DIM, DIM);

        cpu_dists[i] = dist;
        if (dist < cpu_min_dist) {
            cpu_min_dist = dist;
            cpu_min_idx = static_cast<int>(i);
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "[CPU] min_idx: " << cpu_min_idx << ", min_dist: " << cpu_min_dist << ", label(min_idx): " << labels[cpu_min_idx] << std::endl;
    std::cout << "CPU distance computation time: " << cpu_time << " ms" << std::endl;

    std::vector<std::pair<float, int>> cpu_dist_idx_pairs;
    for (std::size_t i = 0; i < N; ++i)
        cpu_dist_idx_pairs.emplace_back(cpu_dists[i], i);
    std::sort(cpu_dist_idx_pairs.begin(), cpu_dist_idx_pairs.end());

    int cpu_topk = 10;
    int cpu_correct_label_count = 0;
    std::cout << "[CPU] Top-" << cpu_topk << " nearest vectors:\n";
    for (int i = 0; i < cpu_topk; ++i) {
        int idx = cpu_dist_idx_pairs[i].second;
        std::cout << "  idx: " << idx << ", dist: " << cpu_dist_idx_pairs[i].first << ", label: " << labels[idx] << "\n";
        if (labels[idx] == query_gt_label)
            cpu_correct_label_count++;
    }
    std::cout << "Correct label count in Top-" << cpu_topk << ": " << cpu_correct_label_count << std::endl;

    double sum = 0.0;
    //double min_dist_2 = 100000000, max_dist = -1;
    //size_t count = 0;
    auto pairwise_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            float dist = cpu_distance_l2(db_vectors.data() + i * DIM, db_vectors.data() + j * DIM, DIM);
            // sum += dist;
            // if (dist < min_dist_2) min_dist_2 = dist;
            // if (dist > max_dist) max_dist = dist;
            //++count;
        }
    }
    auto pairwise_end = std::chrono::high_resolution_clock::now();
    double pairwise_time = std::chrono::duration<double, std::milli>(pairwise_end - pairwise_start).count();
    std::cout << "[CPU] Pairwise (all-to-all) distance matrix computed in " << pairwise_time << " ms" << std::endl;

    /*
    double avg = sum / count;
    std::cout << "Pairwise (all-to-all) distance statistics:\n";
    std::cout << "  avg: " << avg << ", min: " << min_dist_2 << ", max: " << max_dist << std::endl;
    */

    return 0;
}