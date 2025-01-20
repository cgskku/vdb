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

#include <sstream>
#include <string>
#include <map>

#define FileFlag 1

void load_csv_data(const std::string& filename, std::vector<float>& h_data, std::vector<std::string>& h_species, std::size_t& N, std::size_t& DIM) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::vector<std::vector<float>> temp_data;
    bool is_header = true;  // First Row -> Header

    while (std::getline(file, line)) {
        if (is_header) {
            is_header = false;
            continue;  // Skip Header
        }

        std::istringstream s(line);
        std::string value;
        std::vector<float> row;
        int col = 0;

        std::string species_name;

        while (std::getline(s, value, ',')) {
            try {
                if (col == 0) {
                    species_name = value;  // first col: species
                } else {
                    row.push_back(std::stof(value));
                }
                col++;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid numeric value in file at column " << col << ": " << value << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        if (!row.empty()) {
            h_species.push_back(species_name);
            temp_data.push_back(row);
        }
    }
    file.close();

    N = temp_data.size();
    DIM = temp_data[0].size();
    h_data.resize(N * DIM);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < DIM; ++j) {
            h_data[i * DIM + j] = temp_data[i][j];
        }
    }

    std::cout << "Loaded " << N << " rows with " << DIM << " dimensions from " << filename << std::endl;
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

    std::string csv_file = argv[1];
    std::size_t K = std::atoi(argv[2]); // Number of clusters
    int MAX_ITER = std::atoi(argv[3]);

    float *d_samples = nullptr, *d_clusterCenters = nullptr;
    float *d_reducedSamples = nullptr, *d_reducedClusterCenters = nullptr;
    int *d_clusterIndices = nullptr, *d_clusterSizes = nullptr;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t maxChunkSize = prop.sharedMemPerBlock / (K * sizeof(float));
    int chunkSize = (int)maxChunkSize - 1; // chunkSize = (chunkSize / prop.warpSize) * prop.warpSize;
    int TPB = std::__bit_floor(chunkSize) > prop.maxThreadsPerBlock / 2 ? prop.maxThreadsPerBlock / 2 : std::__bit_floor(chunkSize);
    TPB = 128;

    std::vector<float> h_samples;
    std::vector<std::string> h_species;
    std::size_t N;
    std::size_t dimension;


    load_csv_data(csv_file, h_samples, h_species, N, dimension);

    int reducedDim = 2;
    std::vector<float> h_reducedSamples(N * reducedDim);
    std::vector<float> h_clusterCenters(K * dimension);
    std::vector<float> h_reducedClusterCenters(K * reducedDim);
    std::vector<float> h_eigenVec, h_meanVec;
    int *h_clusterIndices = (int*)malloc(N * sizeof(int));

    cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMalloc(&d_clusterIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCenters, K * dimension * sizeof(float));
    cudaMalloc(&d_clusterSizes, K * sizeof(int));
    cudaMalloc(&d_reducedSamples, N * reducedDim * sizeof(float));
    cudaMalloc(&d_reducedClusterCenters, K * reducedDim * sizeof(float));

    cudaMemset(d_clusterIndices, -1, N * sizeof(int));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, N - 1);
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < dimension; ++d) {
            h_clusterCenters[k * dimension + d] = h_samples[distribution(generator) * dimension + d];
        }
    }

    PCA(h_samples, h_reducedSamples, h_eigenVec, h_meanVec, N, dimension, reducedDim);
    std::cout << "PCA done" << std::endl;

    cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reducedSamples, h_reducedSamples.data(), N * reducedDim * sizeof(float), cudaMemcpyHostToDevice);

    for(int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter)
    {
        transform_PCA(h_clusterCenters, h_reducedClusterCenters, h_eigenVec, h_meanVec, K, dimension, reducedDim);
        cudaMemcpy(d_reducedClusterCenters, h_reducedClusterCenters.data(), K * reducedDim * sizeof(float), cudaMemcpyHostToDevice);

        // Cluster assignment step
        // launch_kmeans_labeling(d_samples, d_clusterIndices, d_clusterCenters, N, TPB, K, dimension);
        launch_kmeans_labeling(d_reducedSamples, d_clusterIndices, d_reducedClusterCenters, N, TPB, K, reducedDim);
        cudaDeviceSynchronize();

        // Centroid update step
        launch_kmeans_update_center(d_samples, d_clusterIndices, d_clusterCenters, d_clusterSizes, N, TPB, K, dimension);
        cudaDeviceSynchronize();

        cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_clusterCenters.data(), d_clusterCenters, K * dimension * sizeof(float), cudaMemcpyDeviceToHost);
    
        double sse = compute_SSE(h_samples, h_clusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, dimension);
        std::cout << "Iteration " << cur_iter << ": SSE = " << sse << std::endl;
    }
    
#if FileFlag
    file_write_sample(h_clusterCenters, K, dimension, "centroid.txt");
    file_write_sample(h_reducedClusterCenters, K, reducedDim, "centroid_reduced.txt");
#endif

    std::cout << "\nCluster Results:\n";
    std::map<int, std::map<std::string, int>> cluster_species_count;

    for (std::size_t i = 0; i < N; ++i) {
        int cluster_id = h_clusterIndices[i];
        cluster_species_count[cluster_id][h_species[i]]++;
    }

    for (const auto& cluster : cluster_species_count) {
        std::cout << "Cluster " << cluster.first << ":\n";
        for (const auto& species_count : cluster.second) {
            std::cout << "   " << species_count.first << ": " << species_count.second << " samples\n";
        }
    }

    cudaFree(d_samples);
    cudaFree(d_clusterIndices);
    cudaFree(d_clusterCenters);
    cudaFree(d_clusterSizes);
    cudaFree(d_reducedSamples);
    cudaFree(d_reducedClusterCenters);

    free(h_clusterIndices);

    return 0;
}