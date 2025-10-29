#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "lsh.h"  

template<typename VecType = float>
void generate_sample_data(std::vector<VecType>& h_data, std::vector<VecType>& h_clusterCenters, std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = std::numeric_limits<std::size_t>::max()) {
    std::random_device random_device;
    std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? random_device() : static_cast<unsigned int>(seed));

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)1);
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

// bucket distance check
void check_intra_bucket_distance(const std::vector<float>& h_data, const std::vector<int>& labels, int NUM_BUCKETS, std::size_t N, std::size_t DIM, const std::string& tag) {
    std::cout << "\nChecking intra-bucket average distances for " << tag << "...\n";
    for (int bucket_id = 0; bucket_id < NUM_BUCKETS; ++bucket_id) {
        std::vector<int> indices;
        for (std::size_t i = 0; i < N; ++i)
            if (labels[i] == bucket_id)
                indices.push_back(i);

        if (indices.size() < 2) {
            std::cout << tag << " Bucket " << bucket_id << " has too few points.\n";
            continue;
        }

        double total_distance = 0.0;
        int count = 0;
        for (size_t i = 0; i < indices.size(); ++i)
            for (size_t j = i + 1; j < indices.size(); ++j) {
                double dist2 = 0.0;
                for (int d = 0; d < DIM; ++d) {
                    float diff = h_data[indices[i] * DIM + d] - h_data[indices[j] * DIM + d];
                    dist2 += diff * diff;
                }
                total_distance += std::sqrt(dist2);
                count++;
            }

        double avg_distance = total_distance / count;
        std::cout << tag << " Bucket " << bucket_id << ": Avg pairwise distance = " << avg_distance << std::endl;
    }
}

int main() {
    const std::size_t N = 1000;      
    const std::size_t DIM = 1536;      
    const int NUM_BUCKETS = 5;       
    const int K = 25;            
    const int N_PROJ = 10;           
    const int N_PROBES = 8;            

    std::cout << "Generating clustered data..." << std::endl;
    std::vector<float> h_data;
    std::vector<float> h_clusterCenters;
    generate_sample_data(h_data, h_clusterCenters, N, K, DIM);

    std::cout << "Allocating GPU memory..." << std::endl;
    float* d_data = nullptr;
    int* d_coarse_labels = nullptr;
    cudaMalloc(&d_data, N * DIM * sizeof(float));
    cudaMalloc(&d_coarse_labels, N * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Standard LSH
    {
        std::cout << "\nRunning basic LSH coarse clustering..." << std::endl;
        std::vector<float> h_random_proj(DIM);
        std::mt19937 gen(42);
        std::normal_distribution<float> normal(0, 1);
        for (std::size_t i = 0; i < DIM; ++i)
            h_random_proj[i] = normal(gen);

        float* d_random_proj = nullptr;
        cudaMalloc(&d_random_proj, DIM * sizeof(float));
        cudaMemcpy(d_random_proj, h_random_proj.data(), DIM * sizeof(float), cudaMemcpyHostToDevice);

        launch_lsh(d_data, d_coarse_labels, d_random_proj, N, DIM, NUM_BUCKETS);
        cudaDeviceSynchronize();

        std::vector<int> h_lsh_labels(N);
        cudaMemcpy(h_lsh_labels.data(), d_coarse_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        check_intra_bucket_distance(h_data, h_lsh_labels, NUM_BUCKETS, N, DIM, "Basic LSH");

        cudaFree(d_random_proj);
    }

    // Multi-bit LSH 
    {
        std::cout << "\nRunning multi-bit LSH coarse clustering..." << std::endl;
        std::vector<float> h_multi_random_proj(N_PROJ * DIM);
        std::mt19937 gen(123);
        std::normal_distribution<float> normal(0, 1);
        for (int p = 0; p < N_PROJ; ++p)
            for (std::size_t d = 0; d < DIM; ++d)
                h_multi_random_proj[p * DIM + d] = normal(gen);

        float* d_multi_random_proj = nullptr;
        cudaMalloc(&d_multi_random_proj, N_PROJ * DIM * sizeof(float));
        cudaMemcpy(d_multi_random_proj, h_multi_random_proj.data(), N_PROJ * DIM * sizeof(float), cudaMemcpyHostToDevice);

        launch_multi_bit_lsh(d_data, d_coarse_labels, d_multi_random_proj, N, DIM, NUM_BUCKETS, N_PROJ);
        cudaDeviceSynchronize();

        std::vector<int> h_multi_lsh_labels(N);
        cudaMemcpy(h_multi_lsh_labels.data(), d_coarse_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        check_intra_bucket_distance(h_data, h_multi_lsh_labels, NUM_BUCKETS, N, DIM, "Multi-bit LSH");

        cudaFree(d_multi_random_proj);
    }

    // Multi-probe LSH 
    {
        std::cout << "\nRunning multi-probe LSH coarse clustering..." << std::endl;
        std::vector<float> h_probe_random_proj(N_PROJ * DIM);
        std::mt19937 gen(456);
        std::normal_distribution<float> normal(0, 1);
        for (int p = 0; p < N_PROJ; ++p)
            for (std::size_t d = 0; d < DIM; ++d)
                h_probe_random_proj[p * DIM + d] = normal(gen);

        float* d_probe_random_proj = nullptr;
        cudaMalloc(&d_probe_random_proj, N_PROJ * DIM * sizeof(float));
        cudaMemcpy(d_probe_random_proj, h_probe_random_proj.data(), N_PROJ * DIM * sizeof(float), cudaMemcpyHostToDevice);

        launch_multi_probe_lsh(d_data, d_coarse_labels, d_probe_random_proj, N, DIM, NUM_BUCKETS, N_PROJ, N_PROBES);
        cudaDeviceSynchronize();

        std::vector<int> h_probe_labels(N);
        cudaMemcpy(h_probe_labels.data(), d_coarse_labels, N * sizeof(int), cudaMemcpyDeviceToHost);
        check_intra_bucket_distance(h_data, h_probe_labels, NUM_BUCKETS, N, DIM, "Multi-Probe LSH");

        cudaFree(d_probe_random_proj);
    }

    // Random grouping
    {
        std::cout << "\nRunning random grouping for comparison..." << std::endl;
        std::vector<int> h_random_labels(N);
        std::mt19937 random_gen(12345);
        std::uniform_int_distribution<int> random_dist(0, NUM_BUCKETS - 1);
        for (std::size_t i = 0; i < N; ++i)
            h_random_labels[i] = random_dist(random_gen);

        check_intra_bucket_distance(h_data, h_random_labels, NUM_BUCKETS, N, DIM, "Random");
    }


    cudaFree(d_data);
    cudaFree(d_coarse_labels);

    return 0;
}
