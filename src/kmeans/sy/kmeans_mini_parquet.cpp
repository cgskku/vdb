#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <random>
#include <limits>
#include "kmeans_mini.h"

#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>
#include <arrow/api.h>
#include <arrow/io/api.h>

#define FileFlag 0

static std::vector<float> load_parquet_data_multi(const std::vector<std::string>& file_paths,
                                                  std::size_t& num_rows, std::size_t& num_cols) {
    std::vector<float> data;
    num_rows = 0;
    num_cols = 0;

    const int64_t batch_vals = 4096;

    for (const auto& file_path : file_paths)
    {
        try 
        {
            std::unique_ptr<parquet::ParquetFileReader> parquet_reader = parquet::ParquetFileReader::OpenFile(file_path, false);

            auto meta = parquet_reader->metadata();
            int num_row_groups = meta->num_row_groups();

            // 일단 1536 dim의 double data로 가정
            std::size_t file_cols = (meta->schema()->Column(1)->physical_type() == parquet::Type::DOUBLE) ? 1536 : 0;
            if (file_cols == 0) throw std::runtime_error("Expected 1536-dimensional DOUBLE data at Column(1).");
            if (num_cols == 0) num_cols = file_cols;
            else if (num_cols != file_cols) throw std::runtime_error("All files must have the same embedding dimension.");

            // get total rows
            int64_t file_rows = 0;
            for (int rg = 0; rg < num_row_groups; ++rg) file_rows += meta->RowGroup(rg)->num_rows();

            // resize data
            const std::size_t start_row = num_rows;
            data.resize((start_row + static_cast<std::size_t>(file_rows)) * num_cols);

            // read data
            std::size_t dst_linear_base = start_row * num_cols;
            for (int rg = 0; rg < num_row_groups; ++rg) 
            {
                auto rg_reader = parquet_reader->RowGroup(rg);
                int64_t rg_rows = rg_reader->metadata()->num_rows();

                auto column_reader = rg_reader->Column(1);
                auto* double_reader = dynamic_cast<parquet::DoubleReader*>(column_reader.get());
                if (!double_reader) throw std::runtime_error("Failed to cast Column(1) to DoubleReader.");

                // 이 RowGroup의 총 값 개수 = rg_rows * num_cols
                int64_t remain = rg_rows * static_cast<int64_t>(num_cols);
                std::vector<double> vals(static_cast<std::size_t>(std::min(remain, batch_vals)));
                std::vector<int16_t> def; // 필요시 사용 가능하지만 이번엔 생략
                int64_t values_read = 0;

                while (remain > 0) 
                {
                    int64_t ask = std::min(remain, (int64_t)vals.size());
                    // definition/repetition은 이번 스키마 가정에선 미사용(단순 값 열)
                    double_reader->ReadBatch(ask, nullptr, nullptr, vals.data(), &values_read);
                    if (values_read <= 0) 
                    {
                        throw std::runtime_error("ReadBatch returned zero values unexpectedly.");
                    }
                    // 선형 → (row, col) 매핑해서 data에 복사
                    for (int64_t t = 0; t < values_read; ++t) 
                    {
                        data[dst_linear_base + static_cast<std::size_t>(t)] = static_cast<float>(vals[static_cast<std::size_t>(t)]);
                    }
                    dst_linear_base += static_cast<std::size_t>(values_read);
                    remain -= values_read;
                }
            }

            num_rows += static_cast<std::size_t>(file_rows);
            std::cout << "[LOAD] " << file_path << " : rows=" << file_rows
                      << " (accum=" << num_rows << "), dim=" << num_cols << "\n";

        } catch (const std::exception& e) 
        {
            std::cerr << "Error reading Parquet file " << file_path << ": " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Total loaded: " << num_rows << " rows with " << num_cols << " dimensions\n";
    return data;
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

    // ./kmeans_mini_parquet <parquet1> <parquet2> <TPB> <K> <MAX_ITER> <maxPartialN>
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <Parquet file 1> <Parquet file 2> <Threads per block> <K> <Max iterations> <Max partial N>\n";
        return 1;
    }

    const std::string file1 = argv[1];
    const std::string file2 = argv[2];
    const int TPB = std::atoi(argv[3]);
    const std::size_t K = static_cast<std::size_t>(std::atoll(argv[4]));
    const int MAX_ITER = std::atoi(argv[5]);
    const std::size_t maxPartialN = static_cast<std::size_t>(std::atoll(argv[6]));

    // load data from parquet files
    auto t_load_start = std::chrono::high_resolution_clock::now();
    std::size_t N = 0, dimension = 0;
    std::vector<float> h_samples = load_parquet_data_multi({file1, file2}, N, dimension);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_elapsed = t_load_end - t_load_start;
    std::cout << "[LOAD] Total data load time: " << load_elapsed.count() << " sec\n";

    float *d_samples = nullptr, *d_clusterCenters = nullptr;
    int *d_clusterIndices = nullptr, *d_clusterSizes = nullptr;

    // Allocate GPU memory
    // 전체 데이터를 한 번에 처리하므로 N만큼 할당
    cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMalloc(&d_clusterIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCenters, K * dimension * sizeof(float));
    cudaMalloc(&d_clusterSizes, K * sizeof(int));

    cudaMemset(d_clusterIndices, -1, N * sizeof(int));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));

    std::vector<float> h_clusterCenters(K * dimension);
    int *h_clusterIndices = (int*)malloc(N * sizeof(int));

    // Initialize centroids randomly from the loaded data
    srand(1234);
    for (std::size_t i = 0; i < K; ++i) {
        std::size_t idx = static_cast<std::size_t>(rand()) % N;
        for (std::size_t d = 0; d < dimension; ++d) {
            h_clusterCenters[i * dimension + d] = h_samples[idx * dimension + d];
        }
    }

    cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    for (int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter)
    {
        // Labeling phase - 전체 데이터를 한 번에 처리
        cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
        launch_kmeans_labeling(d_samples, d_clusterIndices, d_clusterCenters, N, 0, TPB, K, dimension);
        cudaDeviceSynchronize();

        // Update phase - 전체 데이터를 한 번에 처리
        launch_kmeans_update_center(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, 0, TPB, K, dimension);
        cudaDeviceSynchronize();

        cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);

        // 디버깅: 클러스터 크기 확인
        std::vector<int> h_clusterSizes(K);
        cudaMemcpy(h_clusterSizes.data(), d_clusterSizes, K * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Cluster sizes: ";
        for (int k = 0; k < K; ++k) {
            std::cout << h_clusterSizes[k] << " ";
        }
        std::cout << std::endl;

        double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
        std::cout << "Iteration " << cur_iter << ": SSE = " << sse << std::endl;
    }
    cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "K-means execution time: " << elapsed.count() << " ms" << std::endl;

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