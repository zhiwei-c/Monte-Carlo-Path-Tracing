#pragma once

#include <string>

#include "../tensor/tensor.cuh"

namespace ior_lut
{
    bool LookupDielectricIor(const std::string &name, float *ior);

    bool LookupConductorIor(const std::string &name, Vec3 *eta, Vec3 *k);
} // namespace ior_lut
