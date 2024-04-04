#ifndef CSRT__RENDERER__BSDF__MICROFACET_HPP
#define CSRT__RENDERER__BSDF__MICROFACET_HPP

#include "../../tensor.hpp"

namespace csrt
{

QUALIFIER_D_H void SampleGgx(const float xi_0, const float xi_1,
                             const float roughness, Vec3 *vec, float *pdf);
QUALIFIER_D_H void SampleGgx(const float xi_0, const float xi_1,
                             const float roughness_u, const float roughness_v,
                             Vec3 *vec, float *pdf);

QUALIFIER_D_H float PdfGgx(const float roughness, const Vec3 &vec);
QUALIFIER_D_H float PdfGgx(const float roughness_u, const float roughness_v,
                           const Vec3 &vec);

QUALIFIER_D_H float SmithG1Ggx(const float roughness, const Vec3 &v,
                               const Vec3 &h);
QUALIFIER_D_H float SmithG1Ggx(const float roughness_u, const float roughness_v,
                               const Vec3 &v, const Vec3 &h);

template <typename T>
QUALIFIER_D_H T FresnelSchlick(const float cos_theta, const T &relectivity)
{
    return (1.0f - relectivity) * static_cast<float>(pow(1.0f - cos_theta, 5)) +
           relectivity;
}

} // namespace csrt

#endif