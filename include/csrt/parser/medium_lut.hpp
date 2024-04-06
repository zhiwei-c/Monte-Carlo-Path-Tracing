#ifndef CSRT__PARSER__MEDIUM_LUT_HPP
#define CSRT__PARSER__MEDIUM_LUT_HPP

#include "../tensor.hpp"

#include <string>

namespace csrt
{

namespace medium_lut
{
    bool LookupIsotropicHomogeneousMedium(const std::string &name,
                                          Vec3 *sigma_a, Vec3 *sigma_s);

    bool LookupHomogeneousMedium(const std::string &name, Vec3 *sigma_a,
                                 Vec3 *sigma_s, Vec3 *g);
} // namespace medium_lut

} // namespace csrt

#endif