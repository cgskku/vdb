#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <limits>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "include/kmeans.h"

#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>
#include <arrow/api.h>
#include <arrow/io/api.h>

#define FileFlag 0

static std::vector<float> load_parquet_data_multi(const std::vector<std::string>& file_paths, std::size_t& num_rows, std::size_t& num_cols) {
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

            // @@ 일단 1536 dim의 double data로 가정
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

                // rowgroup total values = rg_rows * num_cols
                int64_t remain = rg_rows * static_cast<int64_t>(num_cols);
                std::vector<double> vals(static_cast<std::size_t>(std::min(remain, batch_vals)));
                std::vector<int16_t> def; // @@
                int64_t values_read = 0;

                while (remain > 0) 
                {
                    int64_t ask = std::min(remain, (int64_t)vals.size());
                    // @@
                    double_reader->ReadBatch(ask, nullptr, nullptr, vals.data(), &values_read);
                    if (values_read <= 0) 
                    {
                        throw std::runtime_error("ReadBatch returned zero values unexpectedly.");
                    }
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


double compute_SSE(const std::vector<float>& data, const std::vector<float>& centroids, const std::vector<int>& clusterIndices, std::size_t N, std::size_t K, std::size_t DIM) 
{
    double sse = 0.0;
    for (std::size_t n = 0; n < N; ++n) 
    {
        double squaredDistance = 0.0;
        for (std::size_t d = 0; d < DIM; ++d) 
        {
            float vecDiff  = data[n * DIM + d] - centroids[clusterIndices[n] * DIM + d];
            squaredDistance += vecDiff  * vecDiff ;
        }
        sse += squaredDistance;
    }
    return sse;
}


int main(int argc, char* argv[]) 
{
    // ./kmeans_multi <parquet1> <parquet2> <TPB> <K> <MAX_ITER>
    if (argc != 6) 
    {
        std::cerr << "Usage: " << argv[0]
                  << " <Parquet file 1> <Parquet file 2> <Threads per block> <K> <Max iterations>\n";
        return EXIT_FAILURE;
    }

    const std::string file1 = argv[1];
    const std::string file2 = argv[2];
    const int TPB           = std::atoi(argv[3]);
    const std::size_t K     = static_cast<std::size_t>(std::atoll(argv[4]));
    const int MAX_ITER      = std::atoi(argv[5]);
    const bool use_streams  = 1;
    const bool use_corner_turning = 1;
    const bool use_tiling   = 1;
    const int tile_size     = 10000;

    std::cout << "=== K-means Configuration ===" << std::endl;
    std::cout << "Tiling: " << (use_tiling ? "ON" : "OFF") << std::endl;
    std::cout << "Corner turning: " << (use_corner_turning ? "ON" : "OFF") << std::endl;
    std::cout << "Streams: " << (use_streams ? "ON" : "OFF") << std::endl;
    if (use_tiling) 
    {
        std::cout << "Tile size: " << tile_size << std::endl;
    }
    std::cout << "=============================" << std::endl;

    // load data
    auto t_load_start = std::chrono::high_resolution_clock::now();
    std::size_t N = 0, D = 0;
    std::vector<float> data = load_parquet_data_multi({file1, file2}, N, D);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_elapsed = t_load_end - t_load_start;
    std::cout << "[LOAD] Total data load time: " << load_elapsed.count() << " sec\n";

    // GPU setting
    std::vector<float> h_centroids(K * D);
    std::vector<int>   h_assign(N);

    float* d_datapoints = nullptr;
    float* d_centroids  = nullptr;
    int*   d_assign     = nullptr;
    int*   d_sizes      = nullptr;

    cudaMalloc(&d_datapoints, N * D * sizeof(float));
    cudaMalloc(&d_centroids,  K * D * sizeof(float));
    cudaMalloc(&d_assign,     N * sizeof(int));
    cudaMalloc(&d_sizes,      K * sizeof(int));

    // Pinned memory for faster CPU-GPU transfers (minimal usage)
    int* h_assign_pinned = nullptr;
    float* h_centroids_pinned = nullptr;
    cudaMallocHost(&h_assign_pinned, N * sizeof(int));
    cudaMallocHost(&h_centroids_pinned, K * D * sizeof(float));
    
    // Corner turning: transpose centroids for better memory access
    float* d_centroids_T = nullptr;
    if (use_corner_turning) 
    {
        cudaMalloc(&d_centroids_T, K * D * sizeof(float));
        std::cout << "[MEMORY] Using corner turning with transposed centroids" << std::endl;
    }

    // Streams for asynchronous processing
    cudaStream_t* streams = nullptr;
    int num_streams = 0;
    if (use_streams) 
    {
        int num_tiles = (N + tile_size - 1) / tile_size;
        num_streams = std::min(4, num_tiles); // Use up to 4 streams
        streams = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; ++i) 
        {
            cudaStreamCreate(&streams[i]);
        }
        std::cout << "[STREAMS] Created " << num_streams << " CUDA streams for asynchronous processing" << std::endl;
    }
    
    std::cout << "[MEMORY] Using minimal pinned memory for result transfers" << std::endl;

    cudaMemset(d_assign, -1, N * sizeof(int));
    cudaMemset(d_sizes,   0, K * sizeof(int));

    cudaMemcpy(d_datapoints, data.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

    // initialize centroids
    srand(1234); 
    for (std::size_t i = 0; i < K; ++i) 
    {
        std::size_t idx = static_cast<std::size_t>(rand()) % N;
        for (std::size_t d = 0; d < D; ++d) 
        {
            h_centroids[i * D + d] = data[idx * D + d];
        }
    }
    cudaMemcpy(d_centroids, h_centroids.data(), K * D * sizeof(float), cudaMemcpyHostToDevice);

    // Early stopping parameters (scikit-learn style)
    const double tol = 1e-4;  // Relative tolerance for Frobenius norm (default: 1e-4)
    
    std::vector<float> prev_centroids(K * D);
    bool converged = false;
    int final_iteration = 0;

    // kmeans loop
    auto t_km_start = std::chrono::high_resolution_clock::now();
    for (int it = 1; it <= MAX_ITER; ++it) 
    {
        if (use_tiling) 
        {
            // Tile-based processing (load-manager style)
            int num_tiles = (N + tile_size - 1) / tile_size;
            
            // Labeling (process each tile)
            if (use_corner_turning) 
            {
                // Transpose centroids for corner turning
                if (use_streams)    transpose_centers_stream(d_centroids, d_centroids_T, K, D, TPB, streams[0]);
                else                transpose_centers(d_centroids, d_centroids_T, K, D, TPB);
                cudaDeviceSynchronize();
                
                // Corner turning with tiling
                if (use_streams) 
                {
                    // Process tiles in parallel using streams
                    for (int tile = 0; tile < num_tiles; ++tile) 
                    {
                        int tile_start = tile * tile_size;
                        int stream_id = tile % num_streams;
                        launch_kmeans_labeling_corner_turning_tile_stream(d_datapoints, d_assign, d_centroids_T, N, TPB, K, D, tile_start, tile_size, streams[stream_id]);
                    }
                    // Wait for all streams to complete
                    for (int i = 0; i < num_streams; ++i) 
                    {
                        cudaStreamSynchronize(streams[i]);
                    }
                } 
                else 
                {
                    // Sequential processing
                    for (int tile = 0; tile < num_tiles; ++tile) 
                    {
                        int tile_start = tile * tile_size;
                        launch_kmeans_labeling_corner_turning_tile(d_datapoints, d_assign, d_centroids_T, N, TPB, K, D, tile_start, tile_size);
                    }
                    cudaDeviceSynchronize();
                }
            }
            else 
            {
                // Standard tiling approach
                if (use_streams) 
                {
                    // Process tiles in parallel using streams
                    for (int tile = 0; tile < num_tiles; ++tile) 
                    {
                        int tile_start = tile * tile_size;
                        int stream_id = tile % num_streams;
                        launch_kmeans_labeling_tile_stream(d_datapoints, d_assign, d_centroids, N, TPB, K, D, tile_start, tile_size, streams[stream_id]);
                    }
                    // Wait for all streams to complete
                    for (int i = 0; i < num_streams; ++i) 
                    {
                        cudaStreamSynchronize(streams[i]);
                    }
                } else 
                {
                    // Sequential processing
                    for (int tile = 0; tile < num_tiles; ++tile) 
                    {
                        int tile_start = tile * tile_size;
                        launch_kmeans_labeling_tile(d_datapoints, d_assign, d_centroids, N, TPB, K, D, tile_start, tile_size);
                    }
                    cudaDeviceSynchronize();
                }
            }

            // Update centers (reset and accumulate)
            cudaMemset(d_centroids, 0, K * D * sizeof(float));
            cudaMemset(d_sizes, 0, K * sizeof(int));
            
            if (use_streams) 
            {
                // Process tiles in parallel using streams
                for (int tile = 0; tile < num_tiles; ++tile) 
                {
                    int tile_start = tile * tile_size;
                    int stream_id = tile % num_streams;
                    launch_kmeans_update_center_tile_stream(d_datapoints, d_assign, d_centroids, d_sizes, N, TPB, K, D, tile_start, tile_size, streams[stream_id]);
                }
                // Wait for all streams to complete
                for (int i = 0; i < num_streams; ++i) 
                {
                    cudaStreamSynchronize(streams[i]);
                }
            } 
            else 
            {
                // Sequential processing
                for (int tile = 0; tile < num_tiles; ++tile) 
                {
                    int tile_start = tile * tile_size;
                    launch_kmeans_update_center_tile(d_datapoints, d_assign, d_centroids, d_sizes, N, TPB, K, D, tile_start, tile_size);
                }
                cudaDeviceSynchronize();
            }

            // Average centers
            if (use_streams) {
                launch_kmeans_average_centers_stream(d_centroids, d_sizes, K, D, TPB, streams[0]);
                cudaStreamSynchronize(streams[0]);
            } else {
                launch_kmeans_average_centers(d_centroids, d_sizes, K, D, TPB);
                cudaDeviceSynchronize();
            }
        } 
        else 
        {
            // Standard processing
            launch_kmeans_labeling(d_datapoints, d_assign, d_centroids, N, TPB, K, D);
            cudaDeviceSynchronize();

            launch_kmeans_update_center(d_datapoints, d_assign, d_centroids, d_sizes, N, TPB, K, D);
            cudaDeviceSynchronize();
        }

        // Copy from GPU to pinned memory (faster transfer)
        cudaMemcpy(h_assign_pinned,   d_assign,    N * sizeof(int),      cudaMemcpyDeviceToHost);
        cudaMemcpy(h_centroids_pinned, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Copy from pinned memory to regular host memory
        memcpy(h_assign.data(), h_assign_pinned, N * sizeof(int));
        memcpy(h_centroids.data(), h_centroids_pinned, K * D * sizeof(float));

        const double sse = compute_SSE(data, h_centroids, h_assign, N, K, D);
        
        // Frobenius norm based convergence check (excluding first iteration)
        if (it > 1) 
        {
            // Frobenius norm of the difference in cluster centers
            double frobenius_norm_diff = 0.0;
            for (std::size_t k = 0; k < K; ++k) 
            {
                for (std::size_t d = 0; d < D; ++d) 
                {
                    double diff = h_centroids[k * D + d] - prev_centroids[k * D + d];
                    frobenius_norm_diff += diff * diff;
                }
            }
            frobenius_norm_diff = std::sqrt(frobenius_norm_diff);
            
            // Frobenius norm of the previous cluster centers
            double frobenius_norm_prev = 0.0;
            for (std::size_t k = 0; k < K; ++k) 
            {
                for (std::size_t d = 0; d < D; ++d) 
                {
                    double val = prev_centroids[k * D + d];
                    frobenius_norm_prev += val * val;
                }
            }
            frobenius_norm_prev = std::sqrt(frobenius_norm_prev);
            
            // Relative tolerance check (scikit-learn style)
            double relative_change = (frobenius_norm_prev > 0) ? (frobenius_norm_diff / frobenius_norm_prev) : 0.0;
            
            // early stopping condition check
            if (relative_change < tol) 
            {
                std::cout << "Iteration " << it << ": SSE = " << std::fixed << std::setprecision(3) << sse 
                          << " (Converged: Relative change < " << std::scientific << std::setprecision(1) << tol << ")" << std::endl;
                converged = true;
                final_iteration = it;
                break;
            }
            
            std::cout << "Iteration " << it << ": SSE = " << std::fixed << std::setprecision(3) << sse 
                      << " (Relative change: " << std::scientific << std::setprecision(3) << relative_change << ")" << std::endl;
        } 
        else 
        {
            std::cout << "Iteration " << it << ": SSE = " << std::fixed << std::setprecision(3) << sse << std::endl;
        }
        prev_centroids = h_centroids;
    }
    
    if (!converged) 
    {
        final_iteration = MAX_ITER;
        std::cout << "Reached maximum iterations (" << MAX_ITER << ") without convergence" << std::endl;
    }
    auto t_km_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> km_elapsed = t_km_end - t_km_start;
    
    std::cout << "\n=== K-means Execution Summary ===" << std::endl;
    std::cout << "Final iteration: " << final_iteration << " / " << MAX_ITER << std::endl;
    std::cout << "Convergence status: " << (converged ? "CONVERGED" : "NOT CONVERGED") << std::endl;
    std::cout << "Total execution time: " << std::fixed << std::setprecision(3) << km_elapsed.count() << " ms" << std::endl;
    std::cout << "Average time per iteration: " << std::fixed << std::setprecision(3) << km_elapsed.count() / final_iteration << " ms" << std::endl;
    if (converged) 
    {
        std::cout << "Time saved: " << std::fixed << std::setprecision(3) << (MAX_ITER - final_iteration) * (km_elapsed.count() / final_iteration) << " ms" << std::endl;
    }
    std::cout << "Optimization features used:" << std::endl;
    std::cout << "  - Tiling: " << (use_tiling ? "ON" : "OFF") << std::endl;
    std::cout << "  - Corner turning: " << (use_corner_turning ? "ON" : "OFF") << std::endl;
    std::cout << "  - Pinned memory: ON (minimal usage)" << std::endl;
    std::cout << "  - Streams: " << (use_streams ? "ON" : "OFF") << std::endl;
    if (use_streams) {
        std::cout << "  - Number of streams: " << num_streams << std::endl;
    }
    if (use_tiling) {
        std::cout << "  - Tile size: " << tile_size << std::endl;
        int num_tiles = (N + tile_size - 1) / tile_size;
        std::cout << "  - Number of tiles: " << num_tiles << std::endl;
    }
    std::cout << "=================================" << std::endl;

#if FileFlag
    std::ofstream File("kmeans_result.txt");
    // Write final results to File
    File << "Final Centroids:\n";
    for (std::size_t k = 0; k < K; ++k) 
    {
        File << "Centroid " << k << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_clusterCenters[k * dimension + d] << " ";
        }
        File << "\n";
    }

    File << "\nData Points:\n";
    for (std::size_t i = 0; i < N; ++i) 
    {
        File << "Data Point " << i << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_samples[i * dimension + d] << " ";
        }
        File << " -> Cluster " << h_clusterIndices[i] << "\n";
    }
    File.close();
#endif

    // free memory
    cudaFree(d_datapoints);
    cudaFree(d_centroids);
    cudaFree(d_assign);
    cudaFree(d_sizes);
    
    // free corner turning memory
    if (use_corner_turning) cudaFree(d_centroids_T);

    // free streams
    if (use_streams) 
    {
        for (int i = 0; i < num_streams; ++i) cudaStreamDestroy(streams[i]);
        delete[] streams;
    }
    
    // free pinned memory
    cudaFreeHost(h_assign_pinned);
    cudaFreeHost(h_centroids_pinned);
    return 0;
}
