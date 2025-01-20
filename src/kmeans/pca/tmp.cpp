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

#define FileFlag 1

// Function to generate clustered data
template<typename VecType = float>
void generate_sample_data(std::vector<VecType>& h_data, std::vector<VecType>& h_clusterCenters, std::size_t N, std::size_t K, std::size_t Dim, std::size_t seed = std::numeric_limits<std::size_t>::max()) {
    std::random_device random_device;
    std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? random_device() : static_cast<unsigned int>(seed));

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)1);
    std::uniform_int_distribution<std::size_t> idxUnit(0, K - 1);
    std::normal_distribution<VecType> norm((VecType)0, (VecType)0.025);

    h_data.resize(N * Dim);
    h_clusterCenters.resize(K * Dim);

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t dim = 0; dim < Dim; ++dim) {
            h_clusterCenters[k * Dim + dim] = vecUnit(generator);
        }
    }

    for (std::size_t n = 0; n < N; ++n) {
        std::size_t cur_index = idxUnit(generator);
        for (std::size_t dim = 0; dim < Dim; ++dim) {
            h_data[n * Dim + dim] = h_clusterCenters[cur_index * Dim + dim] + norm(generator);
        }
    }
    return;
}


void file_write_sample(std::vector<float>& h_data, int N, int dimension, std::string filename)
{
    std::ofstream File(filename);
    File << "\n======Data Points======\n";
    for (std::size_t i = 0; i < N; ++i) {
        // File << "Data Point " << i << ": " << std::endl;
        File << "[";
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_data[i * dimension + d] << ", ";
        }
        File << "],";
        File << "\n";
    }
    File.close();
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
    for(int i = 0; i < reducedDim; i++) top_variance += sorted_eigVals[i];

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
    for(int i = 0; i < N; i++){
        for(int d = 0; d<Dim; d++){
            tmp_h_data[i * Dim + d] = h_data[i * Dim + d] - h_meanVec[d];
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

void inverse_PCA(
    std::vector<float>& h_data,
    std::vector<float>& h_reducedData,
    std::vector<float>& h_eigenVec,
    std::vector<float>& h_meanVec,
    int N, int Dim, int reducedDim
)
{
    float *d_in =nullptr, *d_out = nullptr, *d_eigenVec = nullptr;
    cudaMalloc(&d_in,  N * reducedDim * sizeof(float));
    cudaMalloc(&d_out, N * Dim * sizeof(float));
    cudaMalloc(&d_eigenVec, Dim * reducedDim * sizeof(float));

    cudaMemcpy(d_in,  h_reducedData.data(), N * reducedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eigenVec, h_eigenVec.data(), Dim * reducedDim * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha  = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, Dim, reducedDim, &alpha, d_in, N, d_eigenVec, Dim, &beta, d_out, N);

    cublasDestroy(handle);

    float* d_meanVec = nullptr;
    cudaMalloc(&d_meanVec, Dim * sizeof(float));
    cudaMemcpy(d_meanVec, h_meanVec.data(), Dim * sizeof(float), cudaMemcpyHostToDevice);
    int TPB = 128;
    launch_addMean(d_out, d_meanVec, N, Dim, TPB);

    h_data.resize(N * Dim);
    cudaMemcpy(h_data.data(), d_out, N * Dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_eigenVec);
    cudaFree(d_meanVec);
}


int main(int argc, char *argv[])
{
    std::cout.precision(10);

    std::size_t N = std::atoi(argv[1]); // Number of data points
    std::size_t K = std::atoi(argv[2]); // Number of clusters
    int MAX_ITER = std::atoi(argv[3]);
    std::size_t dimension = std::atoi(argv[4]); // Dimension of data points

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t maxChunkSize = prop.sharedMemPerBlock / (K * sizeof(float));
    int chunkSize = (int)maxChunkSize - 1; // chunkSize = (chunkSize / prop.warpSize) * prop.warpSize;
    int TPB = std::__bit_floor(chunkSize) > prop.maxThreadsPerBlock / 2 ? prop.maxThreadsPerBlock / 2 : std::__bit_floor(chunkSize);
    TPB = 256;

    // Generate data
    std::vector<float> h_clusterCenters(K * dimension);
    std::vector<float> h_samples(N * dimension);
    generate_sample_data(h_samples, h_clusterCenters, N, K, dimension);
    std::cout << "generate data done" << std::endl;

    int reducedDim = dimension / 2;
    std::vector<float> h_reducedSamples(N * reducedDim);
    std::vector<float> h_reducedClusterCenters(K * reducedDim);
    std::vector<float> h_inverseClusterCenters(K * dimension);
    std::vector<float> h_eigenVec, h_meanVec;
    PCA(h_samples, h_reducedSamples, h_eigenVec, h_meanVec, N, dimension, reducedDim);
    transform_PCA(h_clusterCenters, h_reducedClusterCenters, h_eigenVec, h_meanVec, K, dimension, reducedDim);
    std::cout << "PCA done" << std::endl;
    
#if FileFlag
    file_write_sample(h_clusterCenters, K, dimension, "centroid.txt");
    file_write_sample(h_reducedClusterCenters, K, reducedDim, "centroid_reduced.txt");
#endif

    inverse_PCA(h_inverseClusterCenters, h_reducedClusterCenters, h_eigenVec, h_meanVec, K, dimension, reducedDim);
#if FileFlag
    file_write_sample(h_inverseClusterCenters, K, dimension, "centroid_inverse.txt");
#endif

    return 0;
}