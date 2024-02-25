#pragma once

#include <cstdint>
#include <limits>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_CUDA
#define QUALIFIER_D_H __device__ __host__
#define QUALIFIER_HOST __host__
#else
#define QUALIFIER_D_H
#define QUALIFIER_HOST
#endif

namespace csrt
{

constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();

constexpr float kAabbErrorBound =
    1.0f + 6.0f * (std::numeric_limits<float>::epsilon() * 0.5f) /
               (1.0f - 3.0f * (std::numeric_limits<float>::epsilon() * 0.5f));

} // namespace csrt