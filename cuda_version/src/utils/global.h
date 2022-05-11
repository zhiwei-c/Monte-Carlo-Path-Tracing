#pragma once

#include <iostream>
#include <curand_kernel.h>

#include "vector.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define CheckCudaErrors(val) CheckCuda((val), #val, __FILE__, __LINE__)
#define PrintExcuError() PrintError(__FUNCTION__, __FILE__, __LINE__)

constexpr auto kPi = static_cast<Float>(3.141592653589793);
constexpr auto kPiInv = static_cast<Float>(1.0 / 3.141592653589793);

constexpr auto kEpsilon = static_cast<Float>(1e-10);
constexpr auto kEpsilonL = static_cast<Float>(1e-2);
constexpr auto kEpsilonPdf = static_cast<Float>(1e-3);
constexpr auto kEpsilonDistance = static_cast<Float>(1e-6);

constexpr auto kUintMax = static_cast<uint>(-1);

constexpr auto kFalse = static_cast<int>(false);
constexpr auto kTrue = static_cast<int>(true);

using vec2 = myvec::vec2;
using vec3 = myvec::vec3;

using uvec2 = myvec::uvec2;
using uvec3 = myvec::uvec3;

using gvec2 = glm::dvec2;
using gvec3 = glm::dvec3;
using gvec4 = glm::dvec4;
using gmat3 = glm::dmat3;
using gmat4 = glm::dmat4;

__device__ inline vec3 RandomVec3(curandState *local_rand_state)
{
    return vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
}

__device__ inline vec2 RandomVec2(curandState *local_rand_state)
{
    return vec2(curand_uniform(local_rand_state), curand_uniform(local_rand_state));
}

inline void PrintError(char const *const func, const char *const file, int const line)
{
    std::cerr << "error at " << file << ":" << line << " inside function '" << func << "' \n";
    cudaDeviceReset();
    exit(1);
}

inline void CheckCuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

constexpr size_t Hash(const char *str)
{
    return (*str ? Hash(str + 1) * 256 : 0) + static_cast<size_t>(*str);
}

constexpr size_t operator"" _hash(const char *str, size_t)
{
    return Hash(str);
}

