#include "gpu_sort.h"

// Keep the executable entrypoint thin so the benchmark driver stays reusable.
int main(int argc, char** argv) {
    return run_gpu_sort_demo(argc, argv);
}
