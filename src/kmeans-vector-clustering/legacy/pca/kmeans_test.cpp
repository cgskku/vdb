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
    std::ofstream File("txt/" + filename);
    // File << "\n======Data Points======\n";
    for (std::size_t i = 0; i < N; ++i) {
        // File << "Data Point " << i << ": " << std::endl;
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_data[i * dimension + d] << ", ";
        }
        File << "\n";
    }
    File.close();
}

void file_write_sample(std::vector<float>& h_data, int* h_clusterIndices ,int N, int dimension, std::string filename)
{
    std::ofstream File("txt/" + filename);
    // File << "\n======Data Points======\n";
    for (std::size_t i = 0; i < N; ++i) {
        // File << "Data Point " << i << ": " << std::endl;
        for (std::size_t d = 0; d < dimension; ++d) {
            File << h_data[i * dimension + d] << ", ";
        }
        File << h_clusterIndices[i];
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
    // column major로 바꾸기
    std::vector<float> h_data_col(N * Dim);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < Dim; j++){
            h_data_col[j * N + i] = h_data[i * Dim + j];
        }
    }
    float* d_X = nullptr;
    cudaMalloc(&d_X, N * Dim * sizeof(float));
    cudaMemcpy(d_X, h_data_col.data(), N * Dim * sizeof(float), cudaMemcpyHostToDevice);

    float* d_X_svd = nullptr;
    cudaMalloc(&d_X_svd, N * Dim * sizeof(float));
    cudaMemcpy(d_X_svd, d_X, N * Dim * sizeof(float), cudaMemcpyDeviceToDevice);

    // singular value
    std::vector<float> h_S(Dim);
    float* d_S = nullptr;
    cudaMalloc(&d_S, Dim * sizeof(float));

    float* d_U = nullptr;
    float* d_VT = nullptr;
    cudaMalloc(&d_VT, Dim * Dim * sizeof(float));

    int workspace = 0;
    int info = 0;
    cusolverDnSgesvd_bufferSize(cusolverHandle, N, Dim, &workspace);

    float* d_workspace = nullptr;
    int* d_info = nullptr;
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_workspace, workspace * sizeof(float));

    cusolverDnSgesvd(cusolverHandle, 'N', 'S',
                        N, Dim,
                        d_X_svd, N,
                        d_S, 
                        d_U, N,  
                        d_VT, Dim, 
                        d_workspace, workspace, nullptr, d_info);

    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(info != 0){
        std::cerr << "cusolverDnSgesvd -> info = " << info << std::endl;
        cudaFree(d_X);
        cudaFree(d_X_svd);
        cudaFree(d_S);
        cudaFree(d_VT);
        cudaFree(d_workspace);
        cudaFree(d_info);
        return;
    }

    cudaMemcpy(h_S.data(), d_S, Dim * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> h_VT(Dim * Dim);
    cudaMemcpy(h_VT.data(), d_VT, Dim * Dim * sizeof(float), cudaMemcpyDeviceToHost);

    h_eigenVec.resize(Dim * reducedDim);
    for(int i = 0; i < reducedDim; i++){
        for(int j = 0; j < Dim; j++){
            h_eigenVec[j + i * Dim] = h_VT[i + j * Dim];
        }
    }

    std::cout << "Singular values: ";
    for(int i = 0; i < Dim; i++){
        std::cout << h_S[i] << " ";
    }
    std::cout << std::endl;

    float* d_V_reduced = nullptr;
    cudaMalloc(&d_V_reduced, Dim * reducedDim * sizeof(float));
    cudaMemcpy(d_V_reduced, h_eigenVec.data(), Dim * reducedDim * sizeof(float), cudaMemcpyHostToDevice);

    float* d_outX = nullptr;
    cudaMalloc(&d_outX, N * reducedDim * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, reducedDim, Dim,
                &alpha, d_X, N, d_V_reduced, Dim,
                &beta, d_outX, N);
    std::vector<float> h_reducedData_col(N * reducedDim);
    cudaMemcpy(h_reducedData_col.data(), d_outX, N * reducedDim * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 다시 column major를 row major로 바꾸기
    h_reducedData.resize(N * reducedDim);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < reducedDim; j++){
            h_reducedData[i * reducedDim + j] = h_reducedData_col[j * N + i];
        }
    }

    float total_variance = 0.0f;
    std::vector<float> eigenvalues(Dim);
    for(int i = 0; i < Dim; i++){
        eigenvalues[i] = (h_S[i] * h_S[i]) / (N - 1);
        total_variance += eigenvalues[i];
    }
    std::cout << "Eigenvalues: ";
    for(int i = 0; i < Dim; i++){
        std::cout << eigenvalues[i] << " ";
    }
    std::cout << "\nExplained variance ratio (first " << reducedDim << " components): ";
    float top_variance = 0.0f;
    for (int i = 0; i < reducedDim; i++) {
        top_variance += eigenvalues[i];
    }
    std::cout << (top_variance / total_variance) * 100 << "%" << std::endl;

    cudaFree(d_X);
    cudaFree(d_X_svd);
    cudaFree(d_S);
    cudaFree(d_VT);
    cudaFree(d_workspace);
    cudaFree(d_info);
    cudaFree(d_V_reduced);
    cudaFree(d_outX);
}

void PCA(
    std::vector<float>& h_data,
    std::vector<float>& h_reducedData,
    std::vector<float>& h_eigenVec,
    std::vector<float>& h_meanVec,
    std::vector<float>& h_stdVec,
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

    h_stdVec.resize(Dim, 0.0f);
    for(int d = 0; d < Dim; d++){
        float sum = 0.0f;
        for (int i = 0; i < N; i++){
            float diff = h_data[i * Dim + d] - h_meanVec[d];
            sum += diff * diff;
        }
        h_stdVec[d] = sqrt(sum / (N - 1));
    }

    for(int i = 0; i < N; i++){
        for (int d = 0; d < Dim; d++){
            tmp_h_data[i * Dim + d] = (h_data[i * Dim + d] - h_meanVec[d]) / h_stdVec[d];
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

int main(int argc, char *argv[])
{
    std::cout.precision(10);

    std::string csv_file = argv[1];
    std::size_t K = std::atoi(argv[2]); // Number of clusters
    int MAX_ITER = std::atoi(argv[3]);

    if(argc < 4){
        std::cerr << "Usage: " << argv[0] << " <csv_file> <K> <MAX_ITER>" << std::endl;
        exit(EXIT_FAILURE);
    }

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
    std::vector<float> h_eigenVec, h_meanVec, h_stdVec;
    int *h_clusterIndices = (int*)malloc(N * sizeof(int));

    cudaMalloc(&d_samples, N * dimension * sizeof(float));
    cudaMalloc(&d_clusterIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCenters, K * dimension * sizeof(float));
    cudaMalloc(&d_clusterSizes, K * sizeof(int));
    cudaMalloc(&d_reducedSamples, N * reducedDim * sizeof(float));
    cudaMalloc(&d_reducedClusterCenters, K * reducedDim * sizeof(float));

    cudaMemset(d_clusterIndices, -1, N * sizeof(int));
    cudaMemset(d_clusterSizes, 0, K * sizeof(int));

    std::cout << "Starting PCA " << dimension << ", " << reducedDim << std::endl;
    PCA(h_samples, h_reducedSamples, h_eigenVec, h_meanVec, h_stdVec, N, dimension, reducedDim);
    std::cout << "PCA done" << std::endl;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, N - 1);
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t d = 0; d < reducedDim; ++d) {
            h_reducedClusterCenters[k * reducedDim + d] = h_reducedSamples[distribution(generator) * reducedDim + d];
        }
    }

    cudaMemcpy(d_samples, h_samples.data(), N * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusterCenters, h_clusterCenters.data(), K * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reducedSamples, h_reducedSamples.data(), N * reducedDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reducedClusterCenters, h_reducedClusterCenters.data(), K * reducedDim * sizeof(float), cudaMemcpyHostToDevice);

    for(int cur_iter = 1; cur_iter <= MAX_ITER; ++cur_iter)
    {
        launch_kmeans_labeling(d_reducedSamples, d_clusterIndices, d_reducedClusterCenters, N, TPB, K, reducedDim);
        cudaDeviceSynchronize();

        // Centroid update step
        launch_kmeans_update_center(d_reducedSamples, d_clusterIndices, d_reducedClusterCenters, d_clusterSizes, N, TPB, K, reducedDim);
        cudaDeviceSynchronize();

        cudaMemcpy(h_clusterIndices, d_clusterIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_reducedClusterCenters.data(), d_reducedClusterCenters, K * reducedDim * sizeof(float), cudaMemcpyDeviceToHost);
    
        double sse = compute_SSE(h_reducedSamples, h_reducedClusterCenters, std::vector<int>(h_clusterIndices, h_clusterIndices + N), N, K, reducedDim);
        std::cout << "Iteration " << cur_iter << ": SSE = " << sse << std::endl;
    }
    
#if FileFlag
    file_write_sample(h_clusterCenters, K, dimension, "centroid.txt");
    file_write_sample(h_reducedClusterCenters, K, reducedDim, "centroid_reduced.txt");
    file_write_sample(h_samples, h_clusterIndices, N, dimension, "samples_1.txt");
    file_write_sample(h_reducedSamples, h_clusterIndices, N, reducedDim, "samples_reduced_1.txt");
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