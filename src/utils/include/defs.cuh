#pragma once

#include <limits>

#include "tensor.cuh"

namespace csrt
{

constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();

constexpr float kAabbErrorBound =
    1.0f + 6.0f * (std::numeric_limits<float>::epsilon() * 0.5f) /
               (1.0f - 3.0f * (std::numeric_limits<float>::epsilon() * 0.5f));

} // namespace csrt
