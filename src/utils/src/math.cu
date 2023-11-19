#include "math.cuh"

#include <cmath>

namespace csrt
{

QUALIFIER_D_H Vec3 RandomVec3(uint32_t *seed)
{
    return {RandomFloat(seed), RandomFloat(seed), RandomFloat(seed)};
}

QUALIFIER_D_H float MisWeight(float pdf1, float pdf2)
{
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return pdf1 / (pdf1 + pdf2);
}

QUALIFIER_D_H Vec2 SampleDiskUniform(const float xi_0, const float xi_1)
{
    const float r1 = 2.0f * xi_0 - 1.0f, r2 = 2.0f * xi_1 - 1.0f;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
     */
    float phi, r;
    if (r1 == 0.0f && r2 == 0.0f)
    {
        r = phi = 0;
    }
    else if (Sqr(r1) > Sqr(r2))
    {
        r = r1;
        phi = kPiDiv4 * (r2 / r1);
    }
    else
    {
        r = r2;
        phi = kPiDiv2 - (r1 / r2) * kPiDiv4;
    }
    return {r * cosf(phi), r * sinf(phi)};
}

QUALIFIER_D_H Vec3 SampleConeUniform(const float cos_cutoff, const float xi_0,
                                     const float xi_1)
{
    const float cos_theta = 1.0f - (1.0f - cos_cutoff) * xi_0,
                phi = 2.0f * kPi * xi_1;
    const float sin_theta = sqrt(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    return Vec3{sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta};
}

QUALIFIER_D_H Vec3 SampleSphereUniform(const float xi_0, const float xi_1)
{
    const float cos_theta = 1.0f - 2.0f * xi_0, phi = k2Pi * xi_1;
    const float sin_theta = sqrtf(1.0f - Sqr(cos_theta));
    return {sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta};
}

QUALIFIER_D_H void SampleHemisCos(const float xi_0, const float xi_1, Vec3 *vec,
                                  float *pdf)
{
    const float cos_theta = sqrtf(xi_0), phi = k2Pi * xi_1;
    const float sin_theta = sqrt(1.0f - Sqr(cos_theta));
    *vec = {sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta};
    *pdf = k1DivPi * cos_theta;
}

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

QUALIFIER_D_H float PdfHemisCos(const Vec3 &vec) { return k1DivPi * vec.z; }

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

QUALIFIER_D_H uint32_t BinarySearch(const uint32_t num, float *cdf,
                                    const float target)
{
    uint32_t begin = 0, end = num, middle;
    while (begin + 1 != end)
    {
        middle = (begin + end) >> 1;
        if (cdf[middle] < target)
            begin = middle;
        else if (cdf[middle] > target)
            end = middle;
        else
            return middle;
    }
    return end;
}

QUALIFIER_D_H bool SolveQuadratic(const float a, const float b, const float c,
                                  float *x0, float *x1)
{
    /* Linear case */
    if (a == 0.0f)
    {
        if (b != 0.0f)
        {
            *x0 = *x1 = -c / b;
            return true;
        }
        return false;
    }

    const float discrim = b * b - 4.0f * a * c;
    /* Leave if there is no solution */
    if (discrim < 0.0f)
        return false;

    /* Numerically stable version of (-b (+/-) sqrt_discrim) / (2 * a)
     *
     * Based on the observation that one solution is always
     * accurate while the other is not. Finds the solution of
     * greater magnitude which does not suffer from loss of
     * precision and then uses the identity x1 * x2 = c / a
     */
    float temp, sqrt_discrim = sqrtf(discrim);
    if (b < 0.0f)
        temp = -0.5f * (b - sqrt_discrim);
    else
        temp = -0.5f * (b + sqrt_discrim);
    *x0 = temp / a;
    *x1 = c / temp;

    /* Return the results so that x0 < x1 */
    if (*x0 > *x1)
    {
        float temp = *x0;
        *x0 = *x1;
        *x1 = temp;
    }
    return true;
}

// (right, up, front)
QUALIFIER_D_H void CartesianToSpherical(Vec3 vec, float *theta, float *phi,
                                        float *r)
{
    if (r != nullptr)
        *r = Length(vec);
    vec = Normalize(vec);
    *theta = acosf(fminf(1.0f, fmaxf(-1.0f, vec.y)));
    if (vec.z == 0 && vec.x == 0)
    {
        *phi = 0;
    }
    else
    {
        *phi = atan2f(vec.z, vec.x);
        if (*phi < 0.0f)
            *phi += 2.0f * kPi;
    }
}

// (right, up, front)
QUALIFIER_D_H Vec3 SphericalToCartesian(const float theta, const float phi,
                                        const float r)
{
    const float sin_theta = sinf(theta);
    return {r * sinf(phi) * sin_theta, r * cosf(theta),
            r * cosf(phi) * sin_theta};
}

QUALIFIER_D_H Vec3 LocalToWorld(const Vec3 &local, const Vec3 &up)
{
    Vec3 C;
    if (sqrt(Sqr(up.x) + Sqr(up.z)) > kEpsilonFloat)
    {
        float len_inv = 1.0f / sqrt(Sqr(up.x) + Sqr(up.z));
        C = {up.z * len_inv, 0, -up.x * len_inv};
    }
    else
    {
        float len_inv = 1.0f / sqrt(Sqr(up.y) + Sqr(up.z));
        C = {0, up.z * len_inv, -up.y * len_inv};
    }
    Vec3 B = Normalize(Cross(C, up));
    return Normalize(local.x * B + local.y * C + local.z * up);
}

} // namespace csrt