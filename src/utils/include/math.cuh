#pragma once

#include <limits>

#include "tensor.cuh"

namespace rt
{

constexpr float kLowestFloat = std::numeric_limits<float>::lowest();
constexpr float kMaxFloat = std::numeric_limits<float>::max();
constexpr uint32_t kMaxUint = std::numeric_limits<uint32_t>::max();

constexpr float kEpsilonFloat = std::numeric_limits<float>::epsilon();
constexpr float kEpsilonDistance = 1e-4f;
constexpr float kEpsilon = 1e-2f;

constexpr float kPi = 3.141592653589793f;
constexpr float k2Pi = 3.141592653589793f * 2.0f;
constexpr float kPiDiv2 = 3.141592653589793f * 0.5f;
constexpr float kPiDiv4 = 3.141592653589793f * 0.25f;
constexpr float k1DivPi = 1.0f / kPi;
constexpr float k1Div2Pi = 1.0f / k2Pi;
constexpr float k1Div4Pi = 1.0f / (4.0f * kPi);

QUALIFIER_D_H constexpr float ToRadians(const float degree)
{
    return degree * 0.01745329251994329576923690768489f;
}

template <uint32_t base>
QUALIFIER_D_H float GetVanDerCorputSequence(uint32_t index)
{
    const float base_inv = 1.0f / base;
    float result = 0.0f, frac = base_inv;
    while (index > 0)
    {
        result += frac * (index % base);
        index *= base_inv;
        frac *= base_inv;
    }
    return result;
}

template <uint32_t cycle>
QUALIFIER_D_H uint32_t Tea(uint32_t v0, uint32_t v1)
{
    uint32_t s0 = 0;
    for (uint32_t n = 0; n < cycle; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

// Return a random sample in the range [0, 1) with a simple Linear Congruential
// Generator.
QUALIFIER_D_H constexpr float RandomFloat(uint32_t *seed)
{
    *seed = *seed * 1664525u + 1013904223u;
    return static_cast<float>(*seed & 0x00ffffff) /
           static_cast<float>(0x01000000u);
}

QUALIFIER_D_H Vec3 RandomVec3(uint32_t *seed);
QUALIFIER_D_H float MisWeight(float pdf1, float pdf2);

QUALIFIER_D_H Vec2 SampleDiskUnifrom(const float xi_0, const float xi_1);
QUALIFIER_D_H Vec3 SampleConeUniform(const float cos_cutoff, const float xi_0,
                                     const float xi_1);
QUALIFIER_D_H void SampleHemisCos(const float xi_0, const float xi_1, Vec3 *vec,
                                  float *pdf);
QUALIFIER_D_H void SampleGgx(const float xi_0, const float xi_1,
                             const float roughness, Vec3 *vec, float *pdf);
QUALIFIER_D_H void SampleGgx(const float xi_0, const float xi_1,
                             const float roughness_u, const float roughness_v,
                             Vec3 *vec, float *pdf);

QUALIFIER_D_H float PdfHemisCos(const Vec3 &vec);
QUALIFIER_D_H float PdfGgx(const float roughness, const Vec3 &vec);
QUALIFIER_D_H float PdfGgx(const float roughness_u, const float roughness_v,
                           const Vec3 &vec);
QUALIFIER_D_H float SmithG1Ggx(const float roughness, const Vec3 &v, const Vec3 &h);
QUALIFIER_D_H float SmithG1Ggx(const float roughness_u, const float roughness_v,
                               const Vec3 &v, const Vec3 &h);

template <typename T>
QUALIFIER_D_H T Sqr(const T &t)
{
    return t * t;
}

template <typename T>
QUALIFIER_D_H T Lerp(const T v1, const T v2, const float t)
{
    return (1.0f - t) * v1 + t * v2;
}

template <typename T>
QUALIFIER_D_H T Lerp(const T *v, const float alpha, const float beta,
                     const float gamma)
{
    return alpha * v[0] + beta * v[1] + gamma * v[2];
}

QUALIFIER_D_H uint32_t BinarySearch(const uint32_t num, float *cdf,
                                    const float target);

QUALIFIER_D_H bool SolveQuadratic(const float a, const float b, const float c,
                                  float *x0, float *x1);

QUALIFIER_D_H void CartesianToSpherical(Vec3 vec, float *theta, float *phi,
                                        float *r);
QUALIFIER_D_H Vec3 SphericalToCartesian(const float theta, const float phi,
                                        const float r);

} // namespace rt