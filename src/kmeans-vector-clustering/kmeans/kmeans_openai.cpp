#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <limits>
#include <cuda_runtime.h>
#include "kmeans.h"
#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>
#include <arrow/api.h>
#include <arrow/io/api.h>

// Function to load Parquet data
std::vector<float> load_parquet_data(const std::string& file_path, std::size_t& num_rows, std::size_t& num_cols) {
    std::vector<float> data;

    try {
        // Open the Parquet file
        std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(file_path, false);

        // Get the number of rows and columns
        auto row_group_reader = parquet_reader->RowGroup(0);
        num_rows = parquet_reader->metadata()->num_rows();
        num_cols = parquet_reader->metadata()->schema()->Column(1)->physical_type() == parquet::Type::DOUBLE ? 1536 : 0;

        if (num_cols == 0) {
            throw std::runtime_error("Expected 1536-dimensional double data in Column 1.");
        }

        // Read the data from Column 1
        auto column_reader = row_group_reader->Column(1);
        auto double_reader = dynamic_cast<parquet::DoubleReader*>(column_reader.get());

        if (!double_reader) {
            throw std::runtime_error("Failed to cast Column 1 to DoubleReader.");
        }

        data.resize(num_rows * num_cols);
        double value;
        int16_t definition_level;
        int64_t values_read = 0;

        for (std::size_t i = 0; i < num_rows; ++i) {
            for (std::size_t j = 0; j < num_cols; ++j) {
                double_reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
                data[i * num_cols + j] = static_cast<float>(value);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading Parquet file: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return data;
}

double computeSSE(const std::vector<float>& data, const std::vector<float>& centroids,
                  const std::vector<int>& assignments, std::size_t N, std::size_t K, std::size_t dimension) {
    double sse = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double dist = 0.0;
        for (std::size_t d = 0; d < dimension; ++d) {
            float diff = data[i * dimension + d] - centroids[assignments[i] * dimension + d];
            dist += diff * diff;
        }
        sse += dist;
    }
    return sse;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <Parquet file path> <Threads per block> <Number of clusters> <Max iterations>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string file_path = argv[1];
    int TPB = std::atoi(argv[2]); // Threads per block
    std::size_t K = std::atoi(argv[3]); // Number of clusters
    int MAX_ITER = std::atoi(argv[4]);

    std::size_t N = 0; // Number of data points
    std::size_t dimension = 0; // Dimension of data points
    std::vector<float> data = load_parquet_data(file_path, N, dimension);

    std::vector<float> h_centroids(K * dimension);
    std::vector<int> h_clust_assn(N);

    float* d_datapoints = nullptr;
    float* d_centroids = nullptr;
    int* d_clust_assn = nullptr;
    int* d_clust_sizes = nullptr;

    cudaMalloc(&d_datapoints, N * dimension * sizeof(float));
    cudaMalloc(&d_centroids, K * dimension * sizeof(float));
    cudaMalloc(&d_clust_assn, N * sizeof(int));
    cudaMalloc(&d_clust_sizes, K * sizeof(int));

    cudaMemset(d_clust_assn, -1, N * sizeof(int));
    cudaMemset(d_clust_sizes, 0, K * sizeof(int));

    cudaMemcpy(d_datapoints, data.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize centroids randomly
    for (std::size_t i = 0; i < K; ++i) {
        std::size_t random_idx = rand() % N;
        for (std::size_t d = 0; d < dimension; ++d) {
            h_centroids[i * dimension + d] = data[random_idx * dimension + d];
        }
    }
    cudaMemcpy(d_centroids, h_centroids.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    for (int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter) {
        launchKMeansClusterAssignment(d_datapoints, d_clust_assn, d_centroids, N, TPB, K, dimension);
        cudaDeviceSynchronize();

        launchKMeansCentroidUpdate(d_datapoints, d_clust_assn, d_centroids, d_clust_sizes, N, TPB, K, dimension);
        cudaDeviceSynchronize();

        cudaMemcpy(h_clust_assn.data(), d_clust_assn, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_centroids.data(), d_centroids, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);

        double sse = computeSSE(data, h_centroids, h_clust_assn, N, K, dimension);
        std::cout << "Iteration " << cur_iter << ": SSE = " << sse << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "K-means execution time: " << elapsed.count() << " ms" << std::endl;

    // Save final results to a text file
    std::ofstream file("kmeans_result.txt");
    file << "Final Centroids:\n";
    for (std::size_t i = 0; i < K; ++i) {
        file << "Centroid " << i << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            file << h_centroids[i * dimension + d] << " ";
        }
        file << "\n";
    }
    /*
    file << "\nData Points and Assignments:\n";
    for (std::size_t i = 0; i < N; ++i) {
        file << "Data Point " << i << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            file << data[i * dimension + d] << " ";
        }
        file << " -> Cluster " << h_clust_assn[i] << "\n";
    }
    */
    file.close();

    cudaFree(d_datapoints);
    cudaFree(d_centroids);
    cudaFree(d_clust_assn);
    cudaFree(d_clust_sizes);

    return 0;
}