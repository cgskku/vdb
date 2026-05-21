#include "gpu_sort.h"

#if GPU_SORT_HAS_CUDA
#include <cuda_runtime.h>

void gpu_sort_cuda_translation_unit_anchor() {
    cudaFree(nullptr);
}
#endif
