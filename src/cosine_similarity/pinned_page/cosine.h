#ifndef COSINE_H
#define COSINE_H

#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>

#ifdef __cplusplus
extern "C" {
#endif

void launch_cosine_similarity(
    const float*    d_samples,
    const float*    d_input,
    float*          d_output,
    int N, int D, int TPB,
    cudaStream_t    stream = 0);

#ifdef __cplusplus
}
#endif

// Function to generate random sample data
template<typename VecType = float>
void generate_sample_data(
    std::vector<VecType>& h_data,
    std::size_t N, std::size_t DIM,
    std::size_t seed = 0) 
{
    std::random_device random_device;
    std::mt19937 generator(seed);

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)0.001);
    std::normal_distribution<VecType> norm((VecType)0, (VecType)0.025);

    h_data.resize(N * DIM);

    for(std::size_t n = 0; n < N; ++n){
        for(std::size_t dim = 0; dim < DIM; ++dim){
            h_data[n * DIM + dim] = vecUnit(generator);
        }
    }

    return;
}

template<typename VecType = float>
void generate_sample_data(
    VecType* h_data,
    std::size_t N, std::size_t DIM,
    std::size_t seed = 0) 
{
    std::random_device random_device;
    std::mt19937 generator(seed);

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)0.001);
    std::normal_distribution<VecType> norm((VecType)0, (VecType)0.025);

    for(std::size_t n = 0; n < N; ++n){
        for(std::size_t dim = 0; dim < DIM; ++dim){
            h_data[n * DIM + dim] = vecUnit(generator);
        }
    }

    return;
}

#endif