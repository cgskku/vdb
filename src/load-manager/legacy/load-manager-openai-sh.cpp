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
#include "load-manager.h"
#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>
#include <arrow/api.h>
#include <arrow/io/api.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "[CUDA Error] %s (code %d) at %s:%d\n",       \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

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
    // std::size_t DIM = std::atoi(argv[3]);
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
    std::size_t N = 0; // Number of data points              500,000
    std::size_t dimension = 0; // Dimension of data points   1536
    std::cout << "Loading Start"  << std::endl;
    auto data_loading_start_time = std::chrono::high_resolution_clock::now();


    std::vector<std::pair<int64_t, std::vector<float>>> data =  load_parquet_id_emb_pairs(file_path, N, dimension);


    auto data_loading_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_loading_time = data_loading_end_time - data_loading_start_time;
    std::cout << "@@@@@@@@@@@@@[Time] Data loading took " << total_loading_time.count() << " seconds.\n";
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
    std::cout << "Allocating memory Start (CPU -> GPU)"  << std::endl;
    data_loading_start_time = std::chrono::high_resolution_clock::now();

    float *d_db_vectors; //, *d_dists;
    cudaError_t err = cudaMalloc(&d_db_vectors, N * dimension * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[Error] cudaMalloc failed for d_db_vectors: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    size_t tile_size = 10000; // for tile-based pairwise distance cal.
    size_t num_tiles = (N + tile_size - 1) / tile_size; // for tile-based pairwise distance cal.
    print_cpu_memory_info(0);

    std::cout << "--------------------------------------------------" << std::endl;
    
    size_t max_tile_rows = std::min(tile_size, N); // 10,000 vs 500,000
    size_t max_tile_cols = std::min(tile_size, N);

    // CPU + pinned
    std::vector<float> flat_data(N * dimension);
    for (std::size_t i = 0; i < N; ++i)
        std::copy(data[i].second.begin(), data[i].second.end(), flat_data.begin() + i * dimension);
    size_t data_bytes = flat_data.size() * sizeof(float);

    std::cout << "[Info] data size: " << flat_data.size() << " (bytes: " << data_bytes << ", MB: " << data_bytes / (1024 * 1024) << ")\n";
    cudaMemcpy(d_db_vectors, flat_data.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice); // gpu: d_db_vectors <- cpu: flat_data


    float* h_col_tile_T = nullptr; // H2D 전송 용 buffer - staging 없이 바로 GPU가 read
    CUDA_CHECK(cudaMallocHost((void**)&h_col_tile_T, dimension * max_tile_cols * sizeof(float)));
    // float* h_tile = nullptr;
    // CUDA_CHECK(cudaMallocHost((void**)&h_tile, max_tile_rows * max_tile_cols * sizeof(float)));
    float* h_tile[2] = {nullptr, nullptr}; // 결과를 D2H로 받아 저장
    CUDA_CHECK(cudaMallocHost((void**)&h_tile[0], max_tile_rows * max_tile_cols * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_tile[1], max_tile_rows * max_tile_cols * sizeof(float)));

    // GPU  
    float* d_col_tile_T = nullptr; // H2D하고, 루프 동안 계속 재사용
    CUDA_CHECK(cudaMalloc(&d_col_tile_T, dimension * max_tile_cols * sizeof(float)));
    // float* d_tile = nullptr;
    // CUDA_CHECK(cudaMalloc(&d_tile, max_tile_rows * max_tile_cols * sizeof(float)));

    float* d_tile[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_tile[0], max_tile_rows * max_tile_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tile[1], max_tile_rows * max_tile_cols * sizeof(float)));

    cudaStream_t io_stream, k_stream, d2h_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&io_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&k_stream,   cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking));

    cudaEvent_t up_evt; // H2D 완료 이벤트
    cudaEventCreateWithFlags(&up_evt, cudaEventDisableTiming);
    cudaEvent_t ker_done_evt[2], d2h_done_evt[2];
    for (int b=0; b<2; ++b) {
        cudaEventCreateWithFlags(&ker_done_evt[b],  cudaEventDisableTiming);
        cudaEventCreateWithFlags(&d2h_done_evt[b],  cudaEventDisableTiming);
    }

    data_loading_end_time = std::chrono::high_resolution_clock::now();
    total_loading_time = data_loading_end_time - data_loading_start_time;
    std::cout << "@@@@@@@@@@@@@[Time] Data loading took " << total_loading_time.count() << " seconds.\n";

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Start tile-based pairwise distance calculation..." << std::endl;
    auto tile_start_time = std::chrono::high_resolution_clock::now();

    // tile-based pairwise distance cal.
    for (size_t tile_j = 0; tile_j < num_tiles; ++tile_j) {
        size_t col_start = tile_j * tile_size;
        size_t tile_cols  = std::min(tile_size, N - col_start);

        for (size_t col = 0; col < tile_cols; ++col) {
            const float* src = flat_data.data() + (col_start + col) * dimension;
            for (size_t d = 0; d < dimension; ++d) {
                h_col_tile_T[d * tile_cols + col] = src[d];
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(d_col_tile_T, h_col_tile_T, dimension * tile_cols * sizeof(float), cudaMemcpyHostToDevice, io_stream));
        cudaEventRecord(up_evt, io_stream);
        cudaStreamWaitEvent(k_stream, up_evt, 0);

        for (size_t tile_i = 0; tile_i < num_tiles; ++tile_i) {
            int p = tile_i & 1;

            size_t row_start = tile_i * tile_size;
            size_t tile_rows = std::min(tile_size, N - row_start);

            dim3 block(16, 16);
            dim3 grid((tile_cols + block.x - 1) / block.x, (tile_rows + block.y - 1) / block.y);

            cudaStreamWaitEvent(k_stream, d2h_done_evt[p], 0);
            launch_pairwise_distance_tile_kernel_transpose_stream(d_db_vectors, d_col_tile_T, d_tile[p], (int)N, (int)dimension, (int)row_start, (int)tile_rows, (int)tile_cols, block, grid, k_stream);
            cudaEventRecord(ker_done_evt[p], k_stream);


            cudaStreamWaitEvent(d2h_stream, ker_done_evt[p], 0);
            CUDA_CHECK(cudaMemcpyAsync(h_tile[p], d_tile[p], tile_rows * tile_cols * sizeof(float), cudaMemcpyDeviceToHost, d2h_stream));
            cudaEventRecord(d2h_done_evt[p], d2h_stream);

            if (tile_i == (num_tiles-1) && tile_j == (num_tiles-1)) {
                CUDA_CHECK(cudaEventSynchronize(d2h_done_evt[p]));
                std::cout << "[Tile num_tiles-1,num_tiles-1] pairwise sample:\n";
                for (int r = 0; r < std::min<std::size_t>(7, tile_rows); ++r) {
                    for (int c = 0; c < std::min<std::size_t>(7, tile_cols); ++c)
                        std::cout << h_tile[p][r * tile_cols + c] << " ";
                    std::cout << std::endl;
                }
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(io_stream));
    CUDA_CHECK(cudaStreamSynchronize(k_stream));
    CUDA_CHECK(cudaStreamSynchronize(d2h_stream));
    
    CUDA_CHECK(cudaFree(d_col_tile_T));
    // CUDA_CHECK(cudaFree(d_tile));
    CUDA_CHECK(cudaFreeHost(h_col_tile_T));
    // CUDA_CHECK(cudaFreeHost(h_tile));
    for (int p = 0; p < 2; ++p) {
        CUDA_CHECK(cudaFreeHost(h_tile[p]));
        CUDA_CHECK(cudaFree(d_tile[p]));
    }

    cudaEventDestroy(up_evt);
    cudaStreamDestroy(io_stream);
    cudaStreamDestroy(k_stream);
    cudaStreamDestroy(d2h_stream);
    
    auto tile_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_tile = tile_end_time - tile_start_time;
    std::cout << "@@@@@@@@@@@@@[Time] Tile-based pairwise distance calculation took " << elapsed_tile.count() << " seconds.\n";

    cudaFree(d_db_vectors);

    return 0;
}