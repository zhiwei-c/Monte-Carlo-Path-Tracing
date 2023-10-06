#pragma once

#include <limits>

#include <cstdint>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#define QUALIFIER_DEVICE __device__ __host__
#else
#define QUALIFIER_DEVICE
#endif

constexpr uint64_t kInvalidId = std::numeric_limits<uint64_t>::max();
constexpr float kMaxFloat = std::numeric_limits<float>::max();
constexpr float kLowestFloat = std::numeric_limits<float>::lowest();
constexpr float kEpsilonFloat = std::numeric_limits<float>::epsilon();
