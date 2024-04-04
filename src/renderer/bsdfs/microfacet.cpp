#include "csrt/renderer/bsdfs/microfacet.hpp"

#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H void SampleGgx(const float xi_0, const float xi_1,
                             const float roughness, Vec3 *vec, float *pdf)
{
    const float alpha_2 = Sqr(roughness);
    const float tan_theta_2 = alpha_2 * xi_0 / (1.0f - xi_0), phi = k2Pi * xi_1;
    const float cos_theta = 1.0f / sqrt(1.0f + tan_theta_2),
                sin_theta = sqrt(1.0f - Sqr(cos_theta));
    *vec = {sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta};
    *pdf = 1.0f / (kPi * alpha_2 * pow(cos_theta, 3) *
                   Sqr(1.0f + tan_theta_2 / alpha_2));
}

QUALIFIER_D_H void SampleGgx(const float xi_0, const float xi_1,
                             const float roughness_u, const float roughness_v,
                             Vec3 *vec, float *pdf)
{
    const float phi =
        (atanf(roughness_v / roughness_u * tanf(kPi + k2Pi * xi_1)) +
         kPi * floorf(2.0f * xi_1 + 0.5f));
    const float cos_phi = cosf(phi), sin_phi = sinf(phi),
                alpha_2 = 1.0f / (Sqr(cos_phi / roughness_u) +
                                  Sqr(sin_phi / roughness_v));
    const float tan_theta_2 = alpha_2 * xi_0 / (1.0 - xi_0);
    const float cos_theta = 1.0f / sqrtf(1.0f + tan_theta_2),
                sin_theta = sqrtf(1.0f - Sqr(cos_theta));
    *vec = {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
    *pdf = 1.0f / (kPi * roughness_u * roughness_v * pow(cos_theta, 3) *
                   Sqr(1.0f + tan_theta_2 / alpha_2));
}

QUALIFIER_D_H float PdfGgx(const float roughness, const Vec3 &vec)
{
    const float cos_theta = vec.z;
    if (cos_theta <= 0.0f)
        return 0.0f;
    const float cos_theta_2 = Sqr(cos_theta),
                tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2,
                cos_theta_3 = pow(cos_theta, 3), alpha_2 = Sqr(roughness);
    return alpha_2 / (kPi * cos_theta_3 * Sqr(alpha_2 + tan_theta_2));
}

QUALIFIER_D_H float PdfGgx(const float roughness_u, const float roughness_v,
                           const Vec3 &vec)
{
    const float cos_theta = vec.z;
    if (cos_theta <= 0.0f)
        return 0.0f;
    const float cos_theta_2 = Sqr(cos_theta);
    return cos_theta / (kPi * roughness_u * roughness_v *
                        Sqr(Sqr(vec.x / roughness_u) +
                            Sqr(vec.y / roughness_v) + cos_theta_2));
}

QUALIFIER_D_H float SmithG1Ggx(const float roughness, const Vec3 &v,
                               const Vec3 &h)
{
    const float N_dot_V = v.z;
    if (N_dot_V * h.z <= 0)
        return 0;

    const float cos_theta_2 = Sqr(N_dot_V),
                tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2,
                alpha_2 = Sqr(roughness);

    return 2.0f / (1.0f + sqrtf(1.0 + alpha_2 * tan_theta_2));
}

QUALIFIER_D_H float SmithG1Ggx(const float roughness_u, const float roughness_v,
                               const Vec3 &v, const Vec3 &h)
{
    const float N_dot_V = v.z;
    if (N_dot_V * h.z <= 0)
        return 0;

    const float xy_alpha_2 = Sqr(roughness_u * v.x) + Sqr(roughness_v * v.y),
                tan_theta_2 = xy_alpha_2 / Sqr(N_dot_V);
    return 2.0f / (1.0f + sqrtf(1.0f + tan_theta_2));
}

} // namespace csrt