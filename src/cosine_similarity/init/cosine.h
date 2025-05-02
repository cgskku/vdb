#ifndef COSINE_H
#define COSINE_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_cosine_similarity(
    const float*    d_samples,
    const float*    d_input,
    float*          d_output,             
    int N, int D, int TPB);

#ifdef __cplusplus
}
#endif

#endif