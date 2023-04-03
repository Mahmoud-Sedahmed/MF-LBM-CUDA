#ifndef UTILS_GPU_CUH
#define UTILS_GPU_CUH
#include "externLib_CUDA.cuh"

/**
 * Checks the returned cudaError_t and prints corresponding message in case of error.
 */
#define cudaErrorCheck(ans){ cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif