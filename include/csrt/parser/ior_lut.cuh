#pragma once

#include "../tensor.cuh"

#include <string>

namespace csrt
{

namespace ior_lut
{
    bool LookupDielectricIor(const std::string &name, float *ior);
    bool LookupConductorIor(const std::string &name, Vec3 *eta, Vec3 *k);
} // namespace ior_lut

} // namespace csrt
