#include "csrt/utils/math.hpp"

#include <cmath>

namespace csrt
{

QUALIFIER_D_H float MisWeight(float pdf1, float pdf2)
{
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return pdf1 / (pdf1 + pdf2);
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

QUALIFIER_D_H Mat4 LocalToWorld(const Vec3 &up)
{
    Vec3 C;
    if (sqrt(Sqr(up.x) + Sqr(up.z)) > kEpsilonFloat)
    {
        float len_inv = 1.0f / sqrt(Sqr(up.x) + Sqr(up.z));
        C = {-up.z * len_inv, 0, up.x * len_inv};
    }
    else
    {
        float len_inv = 1.0f / sqrt(Sqr(up.y) + Sqr(up.z));
        C = {0, -up.z * len_inv, up.y * len_inv};
    }
    Vec3 B = Normalize(Cross(C, up));
    return Mat4{{B.x, B.y, B.z, 0},
                {C.x, C.y, C.z, 0},
                {up.x, up.y, up.z, 0},
                {0, 0, 0, 1}};
}

} // namespace csrt