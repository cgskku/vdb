#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <limits>

#include <cuda_runtime.h>
#include "../include/kmeans_mmap.h"   // stream 있는 시그니처 사용: (N, S, TPB, K, dim, cudaStream_t)

#include <sys/mman.h>
#include <unistd.h>

// Parquet/Arrow
#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>
#include <arrow/api.h>
#include <arrow/io/api.h>

// ====== 설정 ======
#define CHUNKSIZE 100000
static constexpr int DIM_ASSUMED = 1536;
// ==================

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  std::cerr<<"CUDA Error: "<<cudaGetErrorString(err)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; \
  std::exit(EXIT_FAILURE);} } while(0)
#endif

// -------------------- 너가 준 멀티 파일 파케 로더 --------------------
static std::vector<float> load_parquet_data_multi(const std::vector<std::string>& file_paths,
                                                  std::size_t& num_rows, std::size_t& num_cols) {
    std::vector<float> data;
    num_rows = 0;
    num_cols = 0;
    const int64_t batch_vals = 4096;

    for (const auto& file_path : file_paths) {
        try {
            std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
                parquet::ParquetFileReader::OpenFile(file_path, false);

            auto meta = parquet_reader->metadata();
            int num_row_groups = meta->num_row_groups();

            // 일단 1536 dim의 double data로 가정
            std::size_t file_cols = (meta->schema()->Column(1)->physical_type() == parquet::Type::DOUBLE) ? DIM_ASSUMED : 0;
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
            for (int rg = 0; rg < num_row_groups; ++rg) {
                auto rg_reader = parquet_reader->RowGroup(rg);
                int64_t rg_rows = rg_reader->metadata()->num_rows();

                auto column_reader = rg_reader->Column(1);
                auto* double_reader = dynamic_cast<parquet::DoubleReader*>(column_reader.get());
                if (!double_reader) throw std::runtime_error("Failed to cast Column(1) to DoubleReader.");

                // 이 RowGroup의 총 값 개수 = rg_rows * num_cols
                int64_t remain = rg_rows * static_cast<int64_t>(num_cols);
                std::vector<double> vals(static_cast<std::size_t>(std::min(remain, batch_vals)));
                int64_t values_read = 0;

                while (remain > 0) {
                    int64_t ask = std::min(remain, (int64_t)vals.size());
                    double_reader->ReadBatch(ask, nullptr, nullptr, vals.data(), &values_read);
                    if (values_read <= 0) {
                        throw std::runtime_error("ReadBatch returned zero values unexpectedly.");
                    }
                    // float로 캐스팅하여 연속 버퍼에 붙여쓰기
                    for (int64_t t = 0; t < values_read; ++t) {
                        data[dst_linear_base + static_cast<std::size_t>(t)] =
                            static_cast<float>(vals[static_cast<std::size_t>(t)]);
                    }
                    dst_linear_base += static_cast<std::size_t>(values_read);
                    remain -= values_read;
                }
            }

            num_rows += static_cast<std::size_t>(file_rows);
            std::cout << "[LOAD] " << file_path << " : rows=" << file_rows
                      << " (accum=" << num_rows << "), dim=" << num_cols << "\n";

        } catch (const std::exception& e) {
            std::cerr << "Error reading Parquet file " << file_path << ": " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    std::cout << "Total loaded: " << num_rows << " rows with " << num_cols << " dimensions\n";
    return data;
}

// -------------------- SSE 계산 (mmap/연속버퍼 직접) --------------------
static double compute_SSE_raw(const float* data, const float* centroids,
                              const int* clusterIndices, std::size_t N, std::size_t K, std::size_t DIM) {
    double sse = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
        int cid = clusterIndices[n];
        if (cid < 0 || (std::size_t)cid >= K) continue;
        const float* x = data + n * DIM;
        const float* c = centroids + (std::size_t)cid * DIM;
        double dsq = 0.0;
        for (std::size_t d = 0; d < DIM; ++d) {
            double diff = (double)x[d] - (double)c[d];
            dsq += diff * diff;
        }
        sse += dsq;
    }
    return sse;
}

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <parquet1> <parquet2> <TPB> <K> <MAX_ITER>\n";
        return 1;
    }

    const std::string file1 = argv[1];
    const std::string file2 = argv[2];
    const int TPB = std::atoi(argv[3]);
    const int K   = std::atoi(argv[4]);
    const int MAX_ITER = std::atoi(argv[5]);

    // 1) Parquet 두 파일 로드 → 연속 float 버퍼
    auto t_load0 = std::chrono::high_resolution_clock::now();
    std::size_t N = 0, D = 0;
    std::vector<float> h_samples = load_parquet_data_multi({file1, file2}, N, D);
    auto t_load1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_sec = t_load1 - t_load0;
    std::cout << "[LOAD] parquet decode time: " << load_sec.count() << " s\n";

    // 2) 익명 mmap에 펼쳐두기(원하면 바로 h_samples.data()를 써도 되지만,
    //    기존 mmap 파이프라인과 동일하게 운용하려고 mmap으로 옮긴다)
    size_t bytes = N * D * sizeof(float);
    void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) { std::cerr << "mmap failed\n"; return 1; }
    float* mappedSample = (float*)mem;
    std::memcpy(mappedSample, h_samples.data(), bytes);
    // (원하면 읽기전용으로 변경) mprotect(mappedSample, bytes, PROT_READ);

    // 3) GPU 리소스
    float *d_centers=nullptr; int *d_indices=nullptr; int *d_sizes=nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_centers, K*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sizes,   K*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_indices, 0xff, N*sizeof(int))); // -1
    CUDA_CHECK(cudaMemset(d_sizes,   0x00, K*sizeof(int)));

    // 초기 중심: 데이터에서 K개 랜덤 샘플
    {
        std::mt19937 rng(1234);
        std::uniform_int_distribution<std::size_t> pick(0, N-1);
        std::vector<float> init(K*D);
        for (int i=0;i<K;++i) {
            std::size_t idx = pick(rng);
            std::memcpy(&init[(size_t)i*D], &mappedSample[idx*D], D*sizeof(float));
        }
        CUDA_CHECK(cudaMemcpy(d_centers, init.data(), K*D*sizeof(float), cudaMemcpyHostToDevice));
    }

    // 4) 핑퐁 버퍼(스트림당 half0/half1 분리) + 스트림/이벤트
    const std::size_t maxn = CHUNKSIZE/2;

    float *h_pinned[2][2];
    float *d_batch [2][2];
    for (int s=0; s<2; ++s) {
        for (int h=0; h<2; ++h) {
            CUDA_CHECK(cudaMallocHost(&h_pinned[s][h], maxn*D*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_batch[s][h],       maxn*D*sizeof(float)));
        }
    }

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking));

    cudaEvent_t done_label[2], done_update[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&done_label[0],  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_label[1],  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_update[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_update[1], cudaEventDisableTiming));

    std::vector<float> h_centers(K*D);
    std::vector<int>   h_indices(N);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter=1; iter<=MAX_ITER; ++iter) {
        // -------- Labeling phase --------
        {
            int chunk_idx = 0;
            std::size_t issued = 0;

            while (issued < N) {
                int s = chunk_idx & 1;
                if (chunk_idx >= 2) CUDA_CHECK(cudaEventSynchronize(done_label[s]));

                std::size_t base = (std::size_t)chunk_idx * CHUNKSIZE;
                if (base >= N) break;

                // half0
                std::size_t n0 = std::min<std::size_t>(maxn, N - base);
                if (n0 > 0) {
                    const float* src0 = &mappedSample[base * D];
                    std::memcpy(h_pinned[s][0], src0, n0*D*sizeof(float));
                    CUDA_CHECK(cudaMemcpyAsync(d_batch[s][0], h_pinned[s][0], n0*D*sizeof(float),
                                               cudaMemcpyHostToDevice, streams[s]));
                    // 포인터 오프셋 + S=0 (커널이 S를 더해 읽든 말든 안전)
                    launch_kmeans_labeling(
                        d_batch[s][0],
                        d_indices + base,
                        d_centers,
                        d_sizes,
                        (int)n0, /*S=*/0, TPB, K, (int)D,
                        streams[s]);
                    issued += n0;
                }

                // half1
                std::size_t n1 = std::min<std::size_t>(maxn, (N > base + n0) ? (N - (base + n0)) : 0);
                if (n1 > 0) {
                    const float* src1 = &mappedSample[(base + n0) * D];
                    std::memcpy(h_pinned[s][1], src1, n1*D*sizeof(float));
                    CUDA_CHECK(cudaMemcpyAsync(d_batch[s][1], h_pinned[s][1], n1*D*sizeof(float),
                                               cudaMemcpyHostToDevice, streams[s]));
                    launch_kmeans_labeling(
                        d_batch[s][1],
                        d_indices + (base + n0),
                        d_centers,
                        d_sizes,
                        (int)n1, /*S=*/0, TPB, K, (int)D,
                        streams[s]);
                    issued += n1;
                }

                CUDA_CHECK(cudaEventRecord(done_label[s], streams[s]));
                ++chunk_idx;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // -------- Update phase --------
        // (누적형 업데이트 커널 전제: 합/카운트를 위해 0으로 초기화)
        CUDA_CHECK(cudaMemset(d_sizes,   0x00, K*sizeof(int)));
        CUDA_CHECK(cudaMemset(d_centers, 0x00, K*D*sizeof(float)));

        {
            int chunk_idx = 0;
            std::size_t issued = 0;

            while (issued < N) {
                int s = chunk_idx & 1;
                if (chunk_idx >= 2) CUDA_CHECK(cudaEventSynchronize(done_update[s]));

                std::size_t base = (std::size_t)chunk_idx * CHUNKSIZE;
                if (base >= N) break;

                // half0
                std::size_t n0 = std::min<std::size_t>(maxn, N - base);
                if (n0 > 0) {
                    const float* src0 = &mappedSample[base * D];
                    std::memcpy(h_pinned[s][0], src0, n0*D*sizeof(float));
                    CUDA_CHECK(cudaMemcpyAsync(d_batch[s][0], h_pinned[s][0], n0*D*sizeof(float),
                                               cudaMemcpyHostToDevice, streams[s]));
                    launch_kmeans_update_center(
                        d_batch[s][0],
                        d_indices + base,
                        d_centers,
                        d_sizes,
                        (int)n0, /*S=*/0, TPB, K, (int)D,
                        streams[s]);
                    issued += n0;
                }

                // half1
                std::size_t n1 = std::min<std::size_t>(maxn, (N > base + n0) ? (N - (base + n0)) : 0);
                if (n1 > 0) {
                    const float* src1 = &mappedSample[(base + n0) * D];
                    std::memcpy(h_pinned[s][1], src1, n1*D*sizeof(float));
                    CUDA_CHECK(cudaMemcpyAsync(d_batch[s][1], h_pinned[s][1], n1*D*sizeof(float),
                                               cudaMemcpyHostToDevice, streams[s]));
                    launch_kmeans_update_center(
                        d_batch[s][1],
                        d_indices + (base + n0),
                        d_centers,
                        d_sizes,
                        (int)n1, /*S=*/0, TPB, K, (int)D,
                        streams[s]);
                    issued += n1;
                }

                CUDA_CHECK(cudaEventRecord(done_update[s], streams[s]));
                ++chunk_idx;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // (필요 시: sum/count → mean 평균화 커널을 여기서 한 번 실행하세요.
        //  당신의 update 커널이 평균까지 해주면 생략)

        // ---- SSE 출력 ----
        CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, N*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_centers.data(), d_centers, K*D*sizeof(float), cudaMemcpyDeviceToHost));
        double sse = compute_SSE_raw(mappedSample, h_centers.data(), h_indices.data(), N, (std::size_t)K, D);
        std::cout << "Iteration " << iter << ": SSE = " << sse << std::endl;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sec = t1 - t0;
    std::cout << "[TIME] total kmeans time: " << sec.count() << " s\n";

    // 정리
    CUDA_CHECK(cudaEventDestroy(done_label[0])); CUDA_CHECK(cudaEventDestroy(done_label[1]));
    CUDA_CHECK(cudaEventDestroy(done_update[0])); CUDA_CHECK(cudaEventDestroy(done_update[1]));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));   CUDA_CHECK(cudaStreamDestroy(streams[1]));
    for (int s=0; s<2; ++s) for (int h=0; h<2; ++h) {
        cudaFree(d_batch[s][h]);
        cudaFreeHost(h_pinned[s][h]);
    }
    cudaFree(d_indices);
    cudaFree(d_centers);
    cudaFree(d_sizes);

    munmap(mappedSample, bytes);
    return 0;
}
