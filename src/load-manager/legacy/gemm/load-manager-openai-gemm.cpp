#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cublas_v2.h>
#include "load-manager.h"
#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>
#include <arrow/api.h>
#include <arrow/io/api.h>

std::vector<std::pair<int64_t, std::vector<float>>> load_parquet_id_emb_pairs(const std::string& file_path, std::size_t& num_rows, std::size_t& num_cols)
{
    std::vector<std::pair<int64_t, std::vector<float>>> data;

    try {
        // 1. Open file
        std::shared_ptr<arrow::io::ReadableFile> infile;
        PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(file_path));
        std::cout << "[Debug] Parquet file open" << std::endl;

        // 2. Create reader
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        PARQUET_ASSIGN_OR_THROW(arrow_reader, parquet::arrow::OpenFile(infile, arrow::default_memory_pool()));
        std::cout << "[Debug] Create Arrow reader" << std::endl;

        // 3. Get schema and indices
        std::shared_ptr<arrow::Schema> schema;
        PARQUET_THROW_NOT_OK(arrow_reader->GetSchema(&schema));
        int id_index = schema->GetFieldIndex("id");
        int emb_index = schema->GetFieldIndex("emb");
        if (id_index == -1 || emb_index == -1) {
            std::cerr << "[Error] 'id' or 'emb' column not found.\n";
            exit(EXIT_FAILURE);
        }

        std::cout << "[Debug] Get column indices (id: " << id_index << ", emb: " << emb_index << ")" << std::endl;

        // 4. Read columns
        std::shared_ptr<arrow::ChunkedArray> id_chunked, emb_chunked;
        PARQUET_THROW_NOT_OK(arrow_reader->ReadColumn(id_index, &id_chunked));
        PARQUET_THROW_NOT_OK(arrow_reader->ReadColumn(emb_index, &emb_chunked));
        std::cout << "[Debug] Read id and emb columns" << std::endl;

        // 5. single chunk for id
        auto id_array = std::static_pointer_cast<arrow::Int64Array>(id_chunked->chunk(0));
        num_rows = id_array->length();

        // 6. Parse all emb chunks
        num_cols = 0;

        for (int c = 0; c < emb_chunked->num_chunks(); ++c) {
            auto list_array = std::static_pointer_cast<arrow::ListArray>(emb_chunked->chunk(c));
            auto value_array = std::static_pointer_cast<arrow::DoubleArray>(list_array->values());

            if (list_array->length() == 0) continue;

            if (num_cols == 0) {
                num_cols = list_array->value_length(0); // assume all rows have same length
                std::cout << "[Debug] Inferred embedding dimension: " << num_cols << std::endl;
            }

            for (int64_t row = 0; row < list_array->length(); ++row) {
                int64_t id = id_array->Value(data.size());
                std::vector<float> emb(num_cols);

                int64_t offset = list_array->value_offset(row);
                for (int64_t d = 0; d < num_cols; ++d) {
                    emb[d] = static_cast<float>(value_array->Value(offset + d));
                }

                data.emplace_back(id, std::move(emb));
            }
        }

        std::cout << "[Debug] Finished reading full id-embedding pairs" << std::endl;
        std::cout << "[Debug] Total rows = " << data.size() << ", dimension = " << num_cols << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to load parquet: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return data;
}

void print_cpu_memory_info(int total_check) {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long mem_total = 0, mem_free = 0, mem_available = 0;
    while (std::getline(meminfo, line)) {
        std::istringstream iss(line);
        std::string key; long val; std::string unit;
        iss >> key >> val >> unit;
        if (key == "MemTotal:") mem_total = val;
        else if (key == "MemAvailable:") mem_available = val;
        else if (key == "MemFree:") mem_free = val;
    }
    if (total_check == 1) std::cout << "[System] CPU Total Memory: " << mem_total / 1024 << " MB\n";
    std::cout << "[System] CPU Available Memory: " << mem_available / 1024 << " MB\n";
}

void print_gpu_memory_info(int total_check) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (total_check == 1) std::cout << "[System] GPU Total Memory: " << total_mem / (1024 * 1024) << " MB\n";
    std::cout << "[System] GPU Available Memory: " << free_mem / (1024 * 1024) << " MB\n";
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " file_path TPB norm_type(l1|l2|cosine)\n";
        return 1;
    }

    std::string file_path = argv[1];
    int TPB = std::atoi(argv[2]); 
    std::string norm_str = argv[3];

    NormType normType;
    if (norm_str == "l1" || norm_str == "L1") normType = L1_NORM;
    else if (norm_str == "l2" || norm_str == "L2") normType = L2_NORM;
    else if (norm_str == "cosine" || norm_str == "COSINE") normType = COSINE_DIST;
    else {
        std::cerr << "Invalid norm type. Use 'l1', 'l2', or 'cosine'.\n";
        return 1;
    }

    print_cpu_memory_info(1);
    print_gpu_memory_info(1);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    std::size_t N = 0; // Number of data points
    std::size_t dimension = 0; // Dimension of data points
    std::cout << "Loading Start"  << std::endl;
    auto data_loading_start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int64_t, std::vector<float>>> data =  load_parquet_id_emb_pairs(file_path, N, dimension);
    auto data_loading_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_loading_time = data_loading_end_time - data_loading_start_time;
    std::cout << "[Time] Data loading took " << total_loading_time.count() << " seconds.\n";
    std::cout << "[Info] Loaded vectors: " << N << " × " << dimension << std::endl;
    print_cpu_memory_info(0);

    // Output only the three vectors from the front
    for (std::size_t i = 0; i < std::min<std::size_t>(3, data.size()); ++i) {
        std::cout << "ID " << data[i].first << ", Embedding: ";
        for (std::size_t j = 0; j < std::min<std::size_t>(5, data[i].second.size()); ++j) {
            std::cout << std::fixed << std::setprecision(4) << data[i].second[j] << " ";
        }
        std::cout << "..." << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;

    float *d_db_vectors;
    cudaError_t err = cudaMalloc(&d_db_vectors, N * dimension * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[Error] cudaMalloc failed for d_db_vectors: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    size_t tile_size = 10000; // for tile-based pairwise distance cal.
    size_t num_tiles = (N + tile_size - 1) / tile_size; // for tile-based pairwise distance cal.
    std::vector<float> h_tile(tile_size * tile_size); // for store results for each tile
    size_t h_tile_bytes = h_tile.size() * sizeof(float);
    std::cout << "[Info] h_tile size: " << h_tile.size() << " (bytes: " << h_tile_bytes << ", MB: " << h_tile_bytes / (1024 * 1024) << ")\n";
    print_cpu_memory_info(0);
    std::cout << "--------------------------------------------------" << std::endl;

    std::vector<float> flat_data(N * dimension);
    for (std::size_t i = 0; i < N; ++i)
        std::copy(data[i].second.begin(), data[i].second.end(), flat_data.begin() + i * dimension);
    size_t data_bytes = flat_data.size() * sizeof(float);
    std::cout << "[Info] data size: " << flat_data.size() << " (bytes: " << data_bytes << ", MB: " << data_bytes / (1024 * 1024) << ")\n";
    cudaMemcpy(d_db_vectors, flat_data.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "--------------------------------------------------" << std::endl;

    // norm2 cal
    float* d_row_norm2_all = nullptr;
    cudaMalloc(&d_row_norm2_all, N * sizeof(float));
    std::cout << "norm2 Cal Start"  << std::endl;
    auto norm2_cal_start_time = std::chrono::high_resolution_clock::now();
    launch_compute_row_norm2_all(d_db_vectors, d_row_norm2_all, (int)N, (int)dimension, TPB);
    auto norm2_cal_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_norm2_cal_time = norm2_cal_end_time - norm2_cal_start_time;
    std::cout << "[Time] norm2 Cal took " << total_norm2_cal_time.count() << " seconds.\n";

    // transpose
    float* d_db_T = nullptr;
    cudaMalloc(&d_db_T, N * dimension * sizeof(float));
    std::cout << "transpose Start"  << std::endl;
    auto transpose_start_time = std::chrono::high_resolution_clock::now();
    launch_transpose_rowmajor_to_colmajor(d_db_vectors, d_db_T, (int)N, (int)dimension);
    auto transpose_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_transpose_time = transpose_end_time - transpose_start_time;
    std::cout << "[Time] transpose took " << total_transpose_time.count() << " seconds.\n";

    cublasHandle_t handle;
    cublasCreate(&handle);
    float* d_C = nullptr; // col-major [tile_rows x tile_cols]
    const float alpha = 1.0, beta = 0.0;

    std::cout << "Start tile-based pairwise distance calculation..." << std::endl;
    auto tile_start_time = std::chrono::high_resolution_clock::now();

    // tile-based pairwise distance cal.
    for (size_t tile_j = 0; tile_j < num_tiles; ++tile_j) {
        size_t col_start = tile_j * tile_size;
        size_t tile_cols  = std::min(tile_size, N - col_start);

        if (d_C) cudaFree(d_C);
        cudaMalloc(&d_C, tile_size * tile_cols * sizeof(float));

        const float* dB = d_db_T + (size_t)col_start * dimension;
        const float* d_col2 = d_row_norm2_all + col_start;
        
        for (size_t tile_i = 0; tile_i < num_tiles; ++tile_i) {
            size_t row_start = tile_i * tile_size;
            size_t tile_rows = std::min(tile_size, N - row_start);

            const float* dA = d_db_T + (size_t)row_start * dimension;

            cublasStatus_t st = cublasSgemm(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)tile_rows, (int)tile_cols, (int)dimension,
                &alpha,
                dA, (int)dimension,
                dB, (int)dimension,
                &beta,
                d_C, (int)tile_rows
            );
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "[Error] cublasSgemm failed, status=" << st << "\n";
                cublasDestroy(handle);
                cudaFree(d_db_T);
                cudaFree(d_row_norm2_all);
                cudaFree(d_db_vectors);
                if (d_C) cudaFree(d_C);
                return 1;
            }

            const float* d_row2 = d_row_norm2_all + row_start;
            launch_fuse_norms_and_dot_to_dist2_colmajor(d_row2, d_col2, d_C, (int)tile_rows, (int)tile_cols, (int)tile_rows);

            size_t bytes = tile_rows * tile_cols * sizeof(float);
            cudaMemcpy(h_tile.data(), d_C, bytes, cudaMemcpyDeviceToHost);
            
            if (tile_i == (num_tiles-1) && tile_j == (num_tiles-1)) {
                std::cout << "[Tile num_tiles-1,num_tiles-1] pairwise sample:\n";
                for (int r = 0; r < std::min<std::size_t>(7, tile_rows); ++r) {
                    for (int c = 0; c < std::min<std::size_t>(7, tile_cols); ++c)
                        std::cout << h_tile[r * tile_cols + c] << " ";
                    std::cout << std::endl;
                }
            }
        }
    }
    
    auto tile_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_tile = tile_end_time - tile_start_time;
    std::cout << "[Time] Tile-based pairwise distance calculation took " << elapsed_tile.count() << " seconds.\n";

    cublasDestroy(handle);
    cudaFree(d_C);
    cudaFree(d_db_T);
    cudaFree(d_row_norm2_all);
    cudaFree(d_db_vectors);

    return 0;
}