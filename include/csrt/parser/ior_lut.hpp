#ifndef CSRT__PARSER__IOR_LUT_HPP
#define CSRT__PARSER__IOR_LUT_HPP

#include "../tensor.hpp"

#include <string>

namespace csrt
{

namespace ior_lut
{
    bool LookupDielectricIor(const std::string &name, float *ior);
    bool LookupConductorIor(const std::string &name, Vec3 *eta, Vec3 *k);
} // namespace ior_lut

} // namespace csrt

#endif
