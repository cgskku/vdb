#include <iostream>
#include <fstream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h> 

#include <algorithm>
#include <chrono>
#include <vector>
#include <random>
#include <limits>
#include "kmeans.h"

#define FileFlag 0

void printGPUstat(cudaDeviceProp prop, int chunkSize, int TPB, int K){
    std::cout << "Device Setting---------------------------------" << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GBs" << std::endl;
    std::cout << "Size of Warp: " << prop.warpSize << std::endl;
    std::cout << "Max Threads Per Block(TPB): " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "chunk size: " << chunkSize << std::endl;
    std::cout << "TPB : " << TPB << std::endl;
    std::cout << "Get Shared Memory : " << K * chunkSize * sizeof(float) << " bytes" <<  std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
}

// Function to generate clustered data
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
            h_data[n * DIM + dim] = h_clusterCenters[cur_index * DIM + dim] + norm(generator);
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

void pca_cuSOLVER(
    std::vector<float>& h_data,
    std::vector<float>& h_reducedData,
    std::vector<float>& h_eigenVec,
    int N, int Dim, int reducedDim,
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t cusolverHandle
    )
{
    float* d_X;
    float* d_outX;

    cudaMalloc(&d_X, N * Dim * sizeof(float));
    cudaMemcpy(d_X, h_data.data(), N * Dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_outX, N * reducedDim * sizeof(float));
    cudaMemset(d_outX, 0, N * reducedDim * sizeof(float));

    size_t sizeCov = Dim * Dim;
    float* d_Cov   = nullptr; // C = X^T * X
    cudaMalloc(&d_Cov, sizeCov * sizeof(float));
    cudaMemset(d_Cov, 0, sizeCov*sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, Dim, Dim, N, &alpha, d_X, N, d_X, N, &beta, d_Cov, Dim);

    float scale = 1.0f / (N - 1);
    cublasSscal(cublasHandle, Dim * Dim, &scale, d_Cov, 1); 

    float* d_eigenVal = nullptr;
    cudaMalloc(&d_eigenVal, Dim * sizeof(float));
    
    int workspace = 0;
    int info = 0;
    cusolverDnSsyevd_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, Dim, d_Cov, Dim, d_eigenVal, &workspace);

    float* d_workspace = nullptr;
    int* d_info = nullptr;
    cudaMalloc(&d_workspace, workspace * sizeof(float));
    cudaMalloc(&d_info, sizeof(int));

    cusolverDnSsyevd(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, Dim, d_Cov, Dim, d_eigenVal, d_workspace, workspace, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(info != 0){
        std::cerr << "cusolverDnSsyevd -> info = " << info << std::endl;
    }

    float* d_Csub = nullptr;
    cudaMalloc(&d_Csub, Dim * reducedDim * sizeof(float));
    
    int TPB = 128;
    launch_extract_top_eigenvectors(d_Cov, d_Csub, Dim, reducedDim, TPB);
    cudaDeviceSynchronize();

    float alpha2 = 1.0f, beta2 = 0.0f;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, reducedDim, Dim, &alpha2, d_X, N, d_Csub, Dim, &beta2, d_outX, N);
    cudaMemcpy(h_reducedData.data(), d_outX, N * reducedDim * sizeof(float), cudaMemcpyDeviceToHost);

    h_eigenVec.resize(Dim * reducedDim);
    cudaMemcpy(h_eigenVec.data(), d_Csub, Dim * reducedDim * sizeof(float), cudaMemcpyDeviceToHost);

    // test
    std::vector<float> h_eigenVal(Dim);
    cudaMemcpy(h_eigenVal.data(), d_eigenVal, Dim * sizeof(float), cudaMemcpyDeviceToHost);

    float total_variance = 0.0f;
    for(auto val : h_eigenVal) total_variance += val;

    std::vector<float> sorted_eigVals = h_eigenVal;
    std::sort(sorted_eigVals.begin(), sorted_eigVals.end(), std::greater<float>());
    float top_variance = 0.0f;
    for(int i = 0; i < reducedDim; ++i) top_variance += sorted_eigVals[i];

    std::cout << "Total Variance: " << total_variance << std::endl;
    std::cout << "Top " << reducedDim << " Variance: " << top_variance << std::endl;
    std::cout << "Explained Variance Ratio: " << (top_variance / total_variance) * 100 << "%" << std::endl;

    cudaFree(d_Cov);
    cudaFree(d_eigenVal);
    cudaFree(d_workspace);
    cudaFree(d_info);
    cudaFree(d_Csub);
    cudaFree(d_X);
    cudaFree(d_outX);
}

void PCA(
    std::vector<float>& h_data,
    std::vector<float>& h_reducedData,
    std::vector<float>& h_eigenVec,
    std::vector<float>& h_meanVec,
    int N, int Dim, int reducedDim)
{
    h_meanVec.resize(Dim, 0.0f);
    std::vector<float> tmp_h_data(N * Dim);

    for(int i = 0; i < N; i++){
        for(int d = 0; d < Dim; d++){
            h_meanVec[d] += h_data[i * Dim + d];
        }
    }
    for(int d = 0; d < Dim; d++){
        h_meanVec[d] /= (float)N;
    }
    for(int i = 0; i < N; i++){
        for(int d = 0; d < Dim; d++){
            tmp_h_data[i * Dim + d] = h_data[i * Dim + d] - h_meanVec[d];
        }
    }

    cublasHandle_t cublasHandle; 
    cusolverDnHandle_t cusolverHandle;

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

    pca_cuSOLVER(tmp_h_data, h_reducedData, h_eigenVec, N, Dim, reducedDim, cublasHandle, cusolverHandle);

    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
}

void transform_PCA(
    std::vector<float>& h_data,
    std::vector<float>& h_reducedData,
    std::vector<float>& h_eigenVec,
    std::vector<float>& h_meanVec,
    int N, int Dim, int reducedDim
)
{
    std::vector<float> tmp_h_data(N * Dim);
    for(int i=0; i < N; i++){
        for(int d=0; d<Dim; d++){
            tmp_h_data[i*Dim + d] = h_data[i*Dim + d] - h_meanVec[d];
        }
    }

    float *d_in =nullptr, *d_out = nullptr, *d_eigenVec = nullptr;
    cudaMalloc(&d_in,  N * Dim * sizeof(float));
    cudaMalloc(&d_out, N * reducedDim * sizeof(float));
    cudaMalloc(&d_eigenVec, Dim * reducedDim * sizeof(float));

    cudaMemcpy(d_in,  tmp_h_data.data(), N * Dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eigenVec, h_eigenVec.data(), Dim * reducedDim * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha  = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, reducedDim, Dim, &alpha, d_in, N, d_eigenVec, Dim, &beta, d_out, N);

    cublasDestroy(handle);

    h_reducedData.resize(N * reducedDim);
    cudaMemcpy(h_reducedData.data(), d_out, N * reducedDim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_eigenVec);
}

int main(int argc, char *argv[])
{
    std::cout.precision(10);

    if (argc != 5) {
        return 1;
    }

    std::size_t N = std::atoi(argv[1]); // Number of data points
    std::size_t K = std::atoi(argv[2]); // Number of clusters
    int MAX_ITER = std::atoi(argv[3]);
    std::size_t dimension = std::atoi(argv[4]); // Dimension of data points
    std::size_t reducedDim = dimension / 2;

    float *d_samples = nullptr, *d_clusterCenters = nullptr;
    float *d_reducedSamples = nullptr, *d_reducedClusterCenters = nullptr;
    int *d_clusterIndices = nullptr, *d_clusterSizes = nullptr;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t maxChunkSize = prop.sharedMemPerBlock / (K * sizeof(float));
    int chunkSize = (int)maxChunkSize - 1;
    // chunkSize = (chunkSize / prop.warpSize) * prop.warpSize;
    // int TPB = std::__bit_floor(chunkSize) > prop.maxThreadsPerBlock / 2 ? prop.maxThreadsPerBlock / 2 : std::__bit_floor(chunkSize);
    int TPB = 128;

    printGPUstat(prop, chunkSize, TPB, K);

    // Allocate GPU memory
    auto mallocStart = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMalloc(&d_clusterIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCenters, K * dimension * sizeof(float));
    cudaMalloc(&d_clusterSizes, K * sizeof(int));
    cudaMalloc(&d_reducedSamples, N * reducedDim * sizeof(float));
    cudaMalloc(&d_reducedClusterCenters, K * reducedDim * sizeof(float));
    auto mallocEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mallocElapsed = mallocEnd - mallocStart;
    std::cout << "Malloc time: " << mallocElapsed.count() << " ms" << std::endl;

    // Initialize GPU memory
    auto memsetStart = std::chrono::high_resolution_clock::now();
    cudaMemset(d_clusterIndices, -1, N * sizeof(int));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));
    auto memsetEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memsetElapsed = memsetEnd - memsetStart;
    std::cout << "Memset time: " << memsetElapsed.count() << " ms" << std::endl;

    // Generate data
    auto generateStart = std::chrono::high_resolution_clock::now();
    std::vector<float> h_clusterCenters(K * dimension), h_samples(N * dimension);
    int *h_clusterIndices = (int*)malloc(N * sizeof(int));
    generate_sample_data(h_samples, h_clusterCenters, N, K, dimension);
    auto generateEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> generateElapsed = generateEnd - generateStart;
    std::cout << "Generate data time: " << generateElapsed.count() << " ms" << std::endl;

    // PCA
    auto pcaStart = std::chrono::high_resolution_clock::now();
    std::vector<float> h_reducedSamples(N * reducedDim);
    std::vector<float> h_reducedClusterCenters(K * reducedDim);
    std::vector<float> h_eigenVec, h_meanVec;
    PCA(h_samples, h_reducedSamples, h_eigenVec, h_meanVec, N, dimension, reducedDim);
    transform_PCA(h_clusterCenters, h_reducedClusterCenters, h_eigenVec, h_meanVec, K, dimension, reducedDim);
    auto pcaEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pcaElapsed = pcaEnd - pcaStart;
    std::cout << "PCA time: " << pcaElapsed.count() << " ms" << std::endl;

    // Copy data to GPU
    auto copyStart = std::chrono::high_resolution_clock::now();
    // cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reducedSamples, h_reducedSamples.data(), N * reducedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reducedClusterCenters, h_reducedClusterCenters.data(), K * reducedDim * sizeof(float), cudaMemcpyHostToDevice);
    auto copyEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copyElapsed = copyEnd - copyStart;
    std::cout << "Copy data time: " << copyElapsed.count() << " ms" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for(int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter)
    {
        // Cluster assignment step
        launch_kmeans_labeling_chunk(d_reducedSamples, d_clusterIndices, d_reducedClusterCenters, N, TPB, K, reducedDim, chunkSize);
        cudaDeviceSynchronize();

        // Centroid update step
        launch_kmeans_update_center_chunk(d_reducedSamples, d_clusterIndices, d_reducedClusterCenters, d_clusterSizes, N, TPB, K, reducedDim, chunkSize);
        cudaDeviceSynchronize();

        if(cur_iter % (MAX_ITER / 10) == 0 || cur_iter == 1)
        {
            cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);
            double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
            std::cout << "Iteration " << cur_iter << " SSE = " << sse << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);
    double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
    std::cout << "SSE = " << sse << std::endl;
    std::cout << "K-means execution time: " << elapsed.count() << " ms" << std::endl;

#if FileFlag
    std::ofstream File("kmeans_result.txt");
    // Write final results to File
    File << "Final Centroids : \n";
    for (std::size_t k = 0; k < K; ++k) {
        File << "Centroid " << k << ": ";
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_clusterCenters[k * dimension + d] << " ";
        }
        File << "\n";
    }

    // File << "\nData Points:\n";
    // for (std::size_t i = 0; i < N; ++i) {
    //     File << "Data Point " << i << ": ";
    //     for (std::size_t d = 0; d < dimension; ++d) {
    //         File << h_samples[i * dimension + d] << " ";
    //     }
    //     File << " -> Cluster " << h_clusterIndices[i] << "\n";
    // }

    File.close();
#endif

    cudaFree(d_samples);
    cudaFree(d_clusterIndices);
    cudaFree(d_clusterCenters);
    cudaFree(d_clusterSizes);
    cudaFree(d_reducedSamples);
    cudaFree(d_reducedClusterCenters);

    free(h_clusterIndices);

    return 0;
}