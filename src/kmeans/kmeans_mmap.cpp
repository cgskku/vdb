#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <string.h>
#include <vector>
#include <random>
#include <limits>
#include "kmeans_mmap.h"
#include <fcntl.h>
#include <unistd.h>

#define CHUNKSIZE 1000000
#define FileFlag 0

template<typename VecType = float>
void generate_sample_data(std::vector<VecType>& h_data, std::vector<VecType>& h_clusterCenters, std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = std::numeric_limits<std::size_t>::max()) {
    std::random_device random_device;
    std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? random_device() : static_cast<unsigned int>(seed));

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)0.001);
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

    if (argc != 6) {
        return 1;
    }

    std::size_t N = std::atoi(argv[1]); // Number of data points
    int TPB = std::atoi(argv[2]); // Threads per block
    std::size_t K = std::atoi(argv[3]); // Number of clusters
    int MAX_ITER = std::atoi(argv[4]);
    std::size_t dimension = std::atoi(argv[5]); // Dimension of data points
    std::size_t maxn = CHUNKSIZE / 2;
    int numOfChunk = (N + CHUNKSIZE - 1) / CHUNKSIZE;
    //std::size_t n = std::atoi(argv[6]);

    float *d_clusterCenters = nullptr;
    int *d_clusterIndices = nullptr, *d_clusterSizes = nullptr;
    float *d_samples = nullptr;

    std::vector<float> h_clusterCenters(K * dimension), h_samples(CHUNKSIZE * dimension);

    const char *sampleFile = "sampleFile";
    const char *centroidsFile = "centroidFile";

    int sampleFd = open(sampleFile, O_RDONLY);
    if (sampleFd < 0)
    {
        printf("open sample file failed\n");
        return 1;
    }

    int centFd = open(centroidsFile, O_RDONLY);
    if (centFd < 0)
    {
        printf("open centroid files failed\n");
        return 1;
    }

    int bytes_read = read(centFd, h_clusterCenters.data(), K * dimension * sizeof(float));
    if (bytes_read < 0)
    {
        printf("read centroid file failed\n");
        return 1;
    }

    int *h_clusterIndices = (int*)malloc(N * sizeof(int));
    // read file from sampleFile
    //generate_sample_data(h_samples, h_clusterCenters, N, K, dimension);

    //cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMalloc(&d_clusterIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCenters, K * dimension * sizeof(float));
    cudaMalloc(&d_clusterSizes, K * sizeof(int));

    cudaMemset(d_clusterIndices, -1, N * sizeof(int));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));

    cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);

    //cudaMalloc(&d_samples, N * dimension * sizeof(float));


    float *d_samples_div[2];
    cudaMalloc(&d_samples_div[0], maxn * dimension * sizeof(float));
    cudaMalloc(&d_samples_div[1], maxn * dimension * sizeof(float));
    float *h_pinned[2];
    cudaMallocHost(&h_pinned[0], maxn * dimension * sizeof(float));
    cudaMallocHost(&h_pinned[1], maxn * dimension * sizeof(float));
    //cudaMallocHost(&h_pinned[2], maxn * dimension * sizeof(float));
    //cudaMallocHost(&h_pinned[3], maxn * dimension * sizeof(float));
    h_samples.clear();

    auto start = std::chrono::high_resolution_clock::now();
    for (int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter)
    {
        std::cout << "iter: " << cur_iter << std::endl; 
        cudaStream_t streams[2];
        for (int i = 0; i < 2; i++)
        {
            cudaStreamCreate(&streams[i]);
        }
        h_samples.resize(CHUNKSIZE * sizeof(float) * dimension);
        h_samples.clear();

        lseek(sampleFd, 0, SEEK_SET);
        cudaMemset(d_clusterSizes, 0, K * sizeof(int));
        for (int i = 0; i < numOfChunk; i++)
        {
            bytes_read = read(sampleFd, h_samples.data(), CHUNKSIZE * dimension * sizeof(float));
            //std::cout << numOfChunk << std::endl;
            if (bytes_read < 0)
            {
                printf("read sample fild failed in %d\n", i);
                return 1;
            }

            for (int j = 0; j < 2; j++)
            {
                int offset = i * CHUNKSIZE + j * maxn;
                int memOffset = j * maxn;
                if (offset >= N)
                    break;
                //int bufferIdx = (i * 2 + j) % 4;
                int n = std::min(maxn, N - offset);  
                memcpy(h_pinned[j], h_samples.data() + memOffset * dimension, n * sizeof(float) * dimension);
                cudaMemcpyAsync(d_samples_div[j], h_pinned[j], n * dimension * sizeof(float), cudaMemcpyHostToDevice, streams[j]);
                launch_kmeans_labeling(d_samples_div[j], d_clusterIndices, d_clusterCenters, d_clusterSizes, n, offset, TPB, K, dimension, streams[j]);
            }
            cudaDeviceSynchronize();
            
        }

        lseek(sampleFd, 0, SEEK_SET);
        cudaMemset(d_clusterCenters, 0, K * dimension * sizeof(float));
        for (int i = 0; i < numOfChunk; i++)
        {
            bytes_read = read(sampleFd, h_samples.data(), CHUNKSIZE * dimension * sizeof(float));
            //std::cout << numOfChunk << std::endl;
            if (bytes_read < 0)
            {
                printf("read sample fild failed\n");
                return 1;
            }

            for (int j = 0; j < 2; j++)
            {
                int offset = i * CHUNKSIZE + j * maxn;
                int memOffset = j * maxn;
                if (offset >= N)
                    break;
                
                int n = std::min(maxn, N - offset);
                memcpy(h_pinned[j], h_samples.data() + memOffset * dimension, n * sizeof(float) * dimension);
                cudaMemcpyAsync(d_samples_div[j], h_pinned[j], n * dimension * sizeof(float), cudaMemcpyHostToDevice, streams[j]);
                launch_kmeans_update_center(d_samples_div[j], d_clusterIndices, d_clusterCenters, d_clusterSizes, n, offset, TPB, K, dimension, streams[j]);
            }
            cudaDeviceSynchronize();
        }
        // cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
        // launch_kmeans_update_center(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, 0, TPB, K, dimension, NULL);
        // cudaDeviceSynchronize();
        

        for (int i = 0; i < 2; i++)
        {
            cudaStreamDestroy(streams[i]);
        }

        // lseek(sampleFd, 0, SEEK_SET);
        // h_samples.resize(N * sizeof(float) * dimension);
        // h_samples.clear();
        // bytes_read = read(sampleFd, h_samples.data(), N * dimension * sizeof(float));
        // if (bytes_read < 0)
        // {
        //     printf("read sample fild failed\n");
        //     return 1;
        // }
        // cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);
        // double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
        // std::cout << "Iteration " << cur_iter << ": SSE = " << sse << std::endl;
    }
    //cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "K-means execution time: " << elapsed.count() << " ms" << std::endl;

    for (int i = 0; i < 2; i++)
    {
        cudaFree(d_samples_div[i]);
        cudaFreeHost(h_pinned[i]);
    }
    close(sampleFd);
    close(centFd);

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