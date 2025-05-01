#ifndef LSH_H
#define LSH_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_lsh(float* d_data, int* d_coarse_labels, const float* d_random_proj, int N, int DIM, int num_buckets);
void launch_multi_bit_lsh(const float* d_data, int* d_coarse_labels, const float* d_random_proj, int N, int DIM, int NUM_BUCKETS, int N_PROJ);
void launch_multi_probe_lsh(const float* d_data, int* d_coarse_labels, const float* d_random_proj, int N, int DIM, int NUM_BUCKETS, int N_PROJ, int N_PROBES);

#ifdef __cplusplus
}
#endif

#endif
