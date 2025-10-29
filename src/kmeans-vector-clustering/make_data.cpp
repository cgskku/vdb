#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <limits>

template<typename VecType = float>
void generate_sample_data(std::vector<VecType>& h_data, std::vector<VecType>& h_clusterCenters, std::size_t N, std::size_t K, std::size_t DIM, std::size_t seed = std::numeric_limits<std::size_t>::max()) {
    std::random_device random_device;
    std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? random_device() : static_cast<unsigned int>(seed));

    std::uniform_real_distribution<VecType> vecUnit((VecType)0, (VecType)0.001);
    std::uniform_int_distribution<std::size_t> idxUnit(0, K - 1);
    std::normal_distribution<VecType> norm((VecType)0, (VecType)0.025);

    h_data.resize(N * DIM);
    for (std::size_t n = 0; n < N; ++n) {
        std::size_t cur_index = idxUnit(generator);
        for (std::size_t dim = 0; dim < DIM; ++dim) {
            h_data[n * DIM + dim] = h_clusterCenters[cur_index * dim + dim] + norm(generator);
        }
    }
    return;
}

template<typename VecType = float>
void generate_centroids(std::vector<VecType>& h_clusterCenters, std::size_t K, std::size_t DIM, std:: size_t seed = std::numeric_limits<std::size_t>::max())
{
    std::random_device random_device;
    std::mt19937 generator(seed == std::numeric_limits<std::size_t>::max() ? random_device() : static_cast<unsigned int>(seed));

    std::uniform_real_distribution<float> vecUnit((VecType)0, (VecType)0.001);
    std::uniform_int_distribution<std::size_t> idxUnit(0, K - 1);
    std::normal_distribution<VecType> norm((VecType)0, (VecType)0.025);

    h_clusterCenters.resize(K * DIM);

    for (std::size_t k = 0; k < K; k++)
    {
        for (std::size_t dim = 0; dim < DIM; dim++)
        {
            h_clusterCenters[k * DIM + dim] = vecUnit(generator);
        }
    }

    return;
}

int main(int argc, char *argv[] )
{
    std::size_t N = std::atoi(argv[1]);
    std::size_t K = std::atoi(argv[2]);
    std::size_t n = std::atoi(argv[3]);
    std::size_t dimension = std::atoi(argv[4]);

    std::vector<float> h_clusterCenters(K * dimension), h_samples(n * dimension);
    int *h_clusterIndices = (int *)malloc(N * sizeof(int));

    std::cout << 1 << std::endl;
    generate_centroids(h_clusterCenters, K, dimension);

    FILE *centFile = fopen("centroidFile", "wb");
    if (centFile == NULL)
    {
        printf("centorid file open failed\n");
        exit(-1);
    }

    size_t wBytes = fwrite(h_clusterCenters.data(), sizeof(float), K * dimension, centFile);
    if (wBytes != K * dimension)
    {
        printf("Error writing to file\n");
        fclose(centFile);
        exit(-1);
    }

    fclose(centFile);

    FILE *sampleFile = fopen("sampleFile", "wb");
    if (sampleFile == NULL)
    {
        printf("sample file open failed\n");
        exit(-1);
    }

    for (int offset = 0; offset < N; offset += n)
    {
        n = std::min(n, N - offset);
        generate_sample_data(h_samples, h_clusterCenters, n, K, dimension);

        std::cout << h_samples[0] << std::endl;
        size_t wBytes = fwrite(h_samples.data(), sizeof(float), n * dimension, sampleFile);

        if (wBytes != n * dimension)
        {
            printf("Error writing to file\n");
            fclose(sampleFile);
            exit(-1);
        }
    }

    fclose(sampleFile);
    printf("Successfully write files\n");
    return 1;
}