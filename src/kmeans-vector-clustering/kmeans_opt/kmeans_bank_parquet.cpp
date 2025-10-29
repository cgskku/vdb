#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <cmath>
#include "include/kmeans_bank.h"
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

// Corner turning function is now defined in cuda/transpose.cu

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

int main(int argc, char *argv[])
{
    std::cout.precision(10);

    // ./kmeans_bank_parquet <parquet1> <parquet2> <TPB> <K> <MAX_ITER>
    if (argc != 6) 
    {
        std::cerr << "Usage: " << argv[0]
                  << " <Parquet file 1> <Parquet file 2> <Threads per block> <K> <Max iterations>\n";
        return 1;
    }

    const std::string file1 = argv[1];
    const std::string file2 = argv[2];
    const int TPB           = std::atoi(argv[3]);
    const std::size_t K     = static_cast<std::size_t>(std::atoll(argv[4]));
    const int MAX_ITER      = std::atoi(argv[5]);

    // load data
    auto t_load_start = std::chrono::high_resolution_clock::now();
    std::size_t N = 0, D = 0;
    std::vector<float> data = load_parquet_data_multi({file1, file2}, N, D);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_elapsed = t_load_end - t_load_start;
    std::cout << "[LOAD] Total data load time: " << load_elapsed.count() << " sec\n";

    // GPU setting with tiling and streams
    std::vector<float> h_centroids(K * D);
    std::vector<int>   h_assign(N);

    float* d_datapoints = nullptr;
    float* d_centroids  = nullptr;
    int*   d_assign     = nullptr;
    int*   d_sizes      = nullptr;

    // Pinned memory for faster transfers (ping-pong)
    float* h_pinned_datapoints = nullptr;
    int*   h_pinned_assign[2] = {nullptr, nullptr};      // ping-pong
    float* h_pinned_centroids[2] = {nullptr, nullptr};   // ping-pong
    
    cudaMallocHost(&h_pinned_datapoints, N * D * sizeof(float));
    cudaMallocHost(&h_pinned_assign[0], N * sizeof(int));
    cudaMallocHost(&h_pinned_assign[1], N * sizeof(int));
    cudaMallocHost(&h_pinned_centroids[0], K * D * sizeof(float));
    cudaMallocHost(&h_pinned_centroids[1], K * D * sizeof(float));
    
    // Corner turning: Transpose data for better memory access pattern
    float* d_datapoints_transposed = nullptr;
    float* d_centroids_transposed = nullptr;
    cudaMalloc(&d_datapoints_transposed, N * D * sizeof(float));
    cudaMalloc(&d_centroids_transposed, K * D * sizeof(float));

    cudaMalloc(&d_datapoints, N * D * sizeof(float));
    cudaMalloc(&d_centroids,  K * D * sizeof(float));
    cudaMalloc(&d_assign,     N * sizeof(int));
    cudaMalloc(&d_sizes,      K * sizeof(int));

    cudaMemset(d_assign, -1, N * sizeof(int));
    cudaMemset(d_sizes,   0, K * sizeof(int));

    // Create CUDA streams like load-manager (2 streams)
    cudaStream_t copy_stream, k_stream;
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&k_stream, cudaStreamNonBlocking);
    
    // Create events for fine-grained synchronization (like load-manager)
    cudaEvent_t up_evt;
    cudaEventCreateWithFlags(&up_evt, cudaEventDisableTiming);
    
    cudaEvent_t ker_done_evt[2], d2h_done_evt[2]; // ping-pong
    for (int p = 0; p < 2; ++p) 
    {
        cudaEventCreateWithFlags(&ker_done_evt[p], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&d2h_done_evt[p], cudaEventDisableTiming);
    }
    
    cudaEventRecord(d2h_done_evt[0], copy_stream);
    cudaEventRecord(d2h_done_evt[1], copy_stream);

    const size_t tile_size = std::min(50000UL, std::max(10000UL, N / 8));  // Dynamic tile size
    const size_t num_tiles = (N + tile_size - 1) / tile_size;
    
    std::cout << "[TILING] Tile size: " << tile_size << ", Number of tiles: " << num_tiles << std::endl;

    // copy data to pinned memory first, then to GPU using copy_stream (like load-manager)
    memcpy(h_pinned_datapoints, data.data(), N * D * sizeof(float));
    
    // H2D copy using copy_stream
    cudaMemcpyAsync(d_datapoints, h_pinned_datapoints, N * D * sizeof(float), cudaMemcpyHostToDevice, copy_stream);
    cudaEventRecord(up_evt, copy_stream); 
    
    // corner turning
    std::cout << "[CORNER TURNING] Transposing data for optimized memory access..." << std::endl;
    cudaStreamWaitEvent(k_stream, up_evt, 0); // Wait for H2D to complete
    transpose_data(d_datapoints, d_datapoints_transposed, N, D, k_stream);
    cudaStreamSynchronize(k_stream);

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
    
    // corner turning
    transpose_data(d_centroids, d_centroids_transposed, K, D, k_stream);
    cudaStreamSynchronize(k_stream);

    // early stopping parameters, scikit-learn style
    const double tol = 1e-4;  // Relative tolerance for Frobenius norm (default: 1e-4)
    
    std::vector<float> prev_centroids(K * D);
    bool converged = false;
    int final_iteration = 0;

    // kmeans loop with tiling and streams optimization
    auto t_km_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "[LOAD-MANAGER STYLE] Using 2 streams with event-based ping-pong like load-manager" << std::endl;
    
    // ping-pong variables
    int ping = 0, pong = 1;
    
    for (int it = 1; it <= MAX_ITER; it++) 
    {                
        // labeling on k_stream
        for (size_t tile = 0; tile < num_tiles; ++tile) 
        {
            size_t start_idx = tile * tile_size;
            size_t end_idx = std::min(start_idx + tile_size, N);
            size_t current_tile_size = end_idx - start_idx;
            
            // launch labeling on k_stream
            launch_kmeans_labeling_tile(d_datapoints_transposed, d_assign, d_centroids_transposed, 
                                       N, TPB, K, D, start_idx, current_tile_size);
        }
        
        // update centers on k_stream
        cudaMemset(d_centroids_transposed, 0, K * D * sizeof(float));
        cudaMemset(d_sizes, 0, K * sizeof(int));
        
        for (size_t tile = 0; tile < num_tiles; ++tile) 
        {
            size_t start_idx = tile * tile_size;
            size_t end_idx = std::min(start_idx + tile_size, N);
            size_t current_tile_size = end_idx - start_idx;
            
            // update on k_stream
            launch_kmeans_update_center_tile(d_datapoints_transposed, d_assign, d_centroids_transposed, d_sizes, 
                                           N, TPB, K, D, start_idx, current_tile_size);
        }

        // average centers on k_stream
        launch_kmeans_average_centers(d_centroids_transposed, d_sizes, K, D, TPB);
        
        // corner turning
        transpose_data(d_centroids_transposed, d_centroids, D, K, k_stream);
        
        // record kernel completion event
        cudaEventRecord(ker_done_evt[ping], k_stream);
        
        // Copy results
        cudaStreamWaitEvent(copy_stream, ker_done_evt[ping], 0);
        cudaMemcpyAsync(h_pinned_assign[ping], d_assign, N * sizeof(int), cudaMemcpyDeviceToHost, copy_stream);
        cudaMemcpyAsync(h_pinned_centroids[ping], d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);
        cudaEventRecord(d2h_done_evt[ping], copy_stream);
        
        // wait for data transfer to complete before processing results
        cudaEventSynchronize(d2h_done_evt[ping]);
        
        // Copy from pinned memory to regular memory
        memcpy(h_assign.data(), h_pinned_assign[ping], N * sizeof(int));
        memcpy(h_centroids.data(), h_pinned_centroids[ping], K * D * sizeof(float));
        
        // swap ping-pong buffers for next iteration
        std::swap(ping, pong);

        // count clusters per iteration
        std::vector<int> cluster_counts(K, 0);
        for (int i = 0; i < N; ++i) 
        {
            cluster_counts[h_assign[i]]++;
        }
        
        // std::cout << "Iteration " << it << " - Cluster distribution: ";
        // for (int k = 0; k < K; ++k) 
        // {
        //     std::cout << "C" << k << ":" << cluster_counts[k];
        //     if (k < K-1) std::cout << ", ";
        // }
        // std::cout << std::endl;

        const double sse = compute_SSE(data, h_centroids, h_assign, N, K, D);
        std::cout << "Iteration " << it << ": SSE = " << sse << std::endl;
        
        // frobenius norm based convergence check
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
            
            // relative tolerance check
            double relative_change = (frobenius_norm_prev > 0) ? (frobenius_norm_diff / frobenius_norm_prev) : 0.0;
            
            // condition check
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
    std::cout << "=================================" << std::endl;

#if FileFlag
    std::ofstream File("kmeans_result.txt");
    // Write final results to File
    File << "Final Centroids:\n";
    for (std::size_t k = 0; k < K; ++k) 
    {
        File << "Centroid " << k << ": ";
        for (std::size_t d = 0; d < D; ++d) {
            File << h_centroids[k * D + d] << " ";
        }
        File << "\n";
    }

    File << "\nData Points:\n";
    for (std::size_t i = 0; i < N; ++i) 
    {
        File << "Data Point " << i << ": ";
        for (std::size_t d = 0; d < D; ++d) {
            File << data[i * D + d] << " ";
        }
        File << " -> Cluster " << h_assign[i] << "\n";
    }
    File.close();
#endif

    // free memory and streams (like load-manager)
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(k_stream);
    
    // free events (like load-manager)
    cudaEventDestroy(up_evt);
    for (int p = 0; p < 2; ++p) {
        cudaEventDestroy(ker_done_evt[p]);
        cudaEventDestroy(d2h_done_evt[p]);
    }
    
    cudaFreeHost(h_pinned_datapoints);
    cudaFreeHost(h_pinned_assign[0]);
    cudaFreeHost(h_pinned_assign[1]);
    cudaFreeHost(h_pinned_centroids[0]);
    cudaFreeHost(h_pinned_centroids[1]);
    
    cudaFree(d_datapoints);
    cudaFree(d_centroids);
    cudaFree(d_datapoints_transposed);
    cudaFree(d_centroids_transposed);
    cudaFree(d_assign);
    cudaFree(d_sizes);

    return 0;
}
