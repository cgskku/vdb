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
#include <fstream>
#include <sstream>
#include <iomanip>
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
        PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &arrow_reader));
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
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " N TPB K DIM norm_type(l1|l2|cosine)\n";
        return 1;
    }

    std::string file_path = argv[1];
    int TPB = std::atoi(argv[2]); 
    std::size_t K = std::atoi(argv[3]); 
    std::size_t DIM = std::atoi(argv[4]);
    std::string norm_str = argv[5];

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

    print_cpu_memory_info(0);

    std::size_t N = 0; // Number of data points
    std::size_t dimension = DIM; // Dimension of data points
    std::cout << "Loading Start"  << std::endl;
    std::vector<std::pair<int64_t, std::vector<float>>> data =  load_parquet_id_emb_pairs(file_path, N, dimension);
    std::cout << "[Info] Loaded vectors: " << N << " × " << DIM << std::endl;

    // 앞에서 3개 벡터만 출력
    for (std::size_t i = 0; i < std::min<std::size_t>(3, data.size()); ++i) {
        std::cout << "ID " << data[i].first << ", Embedding: ";
        for (std::size_t j = 0; j < std::min<std::size_t>(5, data[i].second.size()); ++j) {
            std::cout << std::fixed << std::setprecision(4) << data[i].second[j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    float *d_db_vectors, *d_dists;
    cudaError_t err = cudaMalloc(&d_db_vectors, N * DIM * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[Error] cudaMalloc failed for d_db_vectors: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }


    cudaMalloc(&d_dists, N * sizeof(float));

    std::vector<float> flat_data(N * dimension);
    for (std::size_t i = 0; i < N; ++i)
        std::copy(data[i].second.begin(), data[i].second.end(), flat_data.begin() + i * dimension);
    size_t data_bytes = flat_data.size() * sizeof(float);
    std::cout << "[Info] data size: " << flat_data.size() << " (bytes: " << data_bytes << ", MB: " << data_bytes / (1024 * 1024) << ")\n";

    cudaMemcpy(d_db_vectors, flat_data.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);

    // gpu
    float *d_pairwise;
    cudaError_t err2 = cudaMalloc(&d_pairwise, N * N * sizeof(float));
    if (err2 != cudaSuccess) {
        std::cerr << "[Error] cudaMalloc failed for d_pairwise: " << cudaGetErrorString(err2) << std::endl;
        return 1;
    }

    std::size_t size_in_bytes = N * N * sizeof(float);
    std::size_t size_in_MB = size_in_bytes / (1024 * 1024);
    std::cout << "d_pairwise size: " << size_in_bytes << " bytes (" << size_in_MB << " MB)" << std::endl;

    int blockX = 32, blockY = 32;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);

    print_gpu_memory_info(0);

    launch_pairwise_distance_kernel(d_db_vectors, d_pairwise, N, DIM, normType, blockX, blockY);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        std::cerr << "[Error] pairwise_distance_kernel launch failed: " << cudaGetErrorString(kernel_err) << std::endl;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR after synchronize] " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);

    float pw_msec = 0;
    cudaEventElapsedTime(&pw_msec, t0, t1);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[GPU] Pairwise (all-to-all) distance matrix computed in " << pw_msec << " ms" << std::endl;
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    std::vector<float> h_pairwise(std::min<size_t>((size_t)N*N, 100));
    cudaMemcpy(h_pairwise.data(), d_pairwise, h_pairwise.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[GPU] Pairwise distance sample (first 3x3):" << std::endl;
    for (int i = 0; i < std::min<int>(3, N); ++i) {
        for (int j = 0; j < std::min<int>(3, N); ++j)
            std::cout << h_pairwise[i*N + j] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_pairwise);
    cudaFree(d_db_vectors); cudaFree(d_dists);
    
    return 0;
}