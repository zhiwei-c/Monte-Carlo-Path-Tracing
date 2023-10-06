#pragma once

#include "../tensor/tensor.cuh"
#include <cassert>

constexpr float kEpsilon = 1e-2f;
constexpr float kEpsilonDistance = 1e-4f;
constexpr float kAabbErrorBound = 1.0f + 6.0f * (std::numeric_limits<float>::epsilon() * 0.5f) / (1.0f - 3.0f * (std::numeric_limits<float>::epsilon() * 0.5f));

constexpr float kPi = 3.141592653589793f;
constexpr float kPiInv = 1.0f / 3.141592653589793f;
constexpr float kTwoPi = 2.0f * 3.141592653589793f;
constexpr float kOneDivTwoPi = 1.0f / (2.0f * 3.141592653589793f);

constexpr int UP_DIM_WORLD = 1;
constexpr int FRONT_DIM_WORLD = 2;
constexpr int RIGHT_DIM_WORLD = 0;

QUALIFIER_DEVICE Vec3 ToLocal(const Vec3 &dir, const Vec3 &up);
QUALIFIER_DEVICE Vec3 ToWorld(const Vec3 &dir, const Vec3 &normal);
QUALIFIER_DEVICE void CartesianToSpherical(Vec3 vec, float *theta, float *phi, float *r);

inline QUALIFIER_DEVICE Vec3 SphericalToCartesian(float theta, float phi, float r)
{
    float sin_theta = sinf(theta);
    return {r * sinf(phi) * sin_theta, r * cosf(theta), r * cosf(phi) * sin_theta};
}

QUALIFIER_DEVICE bool SolveQuadratic(float a, float b, float c, float &x0, float &x1);

inline QUALIFIER_DEVICE float ToRadians(float degrees)
{
    return degrees * 0.01745329251994329576923690768489f;
}

inline QUALIFIER_DEVICE int Modulo(const int a, const int b)
{
    int c = a % b;
    if (c < 0)
        c += b;
    return c;
}

inline QUALIFIER_DEVICE float Lerp(const float v1, const float v2, const float t)
{
    return (1.0f - t) * v1 + t * v2;
}

inline QUALIFIER_DEVICE Vec3 Lerp(const Vec3 &v1, const Vec3 &v2, const float t)
{
    return (1.0f - t) * v1 + t * v2;
}

inline QUALIFIER_DEVICE Vec3 TransfromPoint(const Mat4 &m, const Vec3 &p)
{
    const Vec4 temp = Mul(m, Vec4(p.x, p.y, p.z, 1.0f));
    return {temp.x / temp.w, temp.y / temp.w, temp.z / temp.w};
}

inline QUALIFIER_DEVICE Vec3 TransfromVector(const Mat4 &m, const Vec3 &v)
{
    const Vec4 temp = Mul(m, Vec4(v.x, v.y, v.z, 0.0f));
    return Normalize({temp.x, temp.y, temp.z});
}

inline QUALIFIER_DEVICE uint64_t Tea(uint64_t v0, uint64_t v1, const int cycle)
{
    uint64_t s0 = 0;
    for (uint64_t n = 0; n < cycle; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

inline QUALIFIER_DEVICE float GetVanDerCorputSequence(uint32_t index, const int base)
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

// Generate random float in [0, 1) with a simple linear congruential generator.
inline QUALIFIER_DEVICE float RandomFloat(uint64_t *previous)
{
    *previous = *previous * 1664525u + 1013904223u;
    return static_cast<float>(*previous & 0x00ffffff) / static_cast<float>(0x01000000u);
}

inline QUALIFIER_DEVICE Vec3 SampleConeUniform(float cos_cutoff, float xi_0, float xi_1)
{
    const float cos_theta = 1.0f - (1.0f - cos_cutoff) * xi_0,
                phi = 2.0f * kPi * xi_1,
                sin_theta = sqrt(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    return Vec3{sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta};
}

inline QUALIFIER_DEVICE Vec3 SampleSphereUniform(float xi_0, float xi_1)
{
    const float cos_theta = 1.0f - 2.0f * xi_0,
                phi = 2.0f * kPi * xi_1,
                sin_theta = sqrt(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    return Vec3{sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta};
}

///\brief 均匀抽样半径 1 的圆盘
inline QUALIFIER_DEVICE Vec2 SampleDiskUnifrom(float xi_0, float xi_1)
{
    const float r1 = 2.0f * xi_0 - 1.0f,
                r2 = 2.0f * xi_1 - 1.0f;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
    float phi, r;
    if (r1 == 0.0f && r2 == 0.0f)
    {
        r = phi = 0;
    }
    else if (r1 * r1 > r2 * r2)
    {
        r = r1;
        phi = (kPi * 0.25f) * (r2 / r1);
    }
    else
    {
        r = r2;
        phi = (kPi * 0.5f) - (r1 / r2) * (kPi * 0.25f);
    }
    return {r * cosf(phi), r * sinf(phi)};
}

inline QUALIFIER_DEVICE void SampleHemisCos(float x_1, float x_2, Vec3 &dir, float &pdf)
{
    float cos_theta = sqrt(x_1),
          phi = 2.0f * kPi * x_2,
          sin_theta = sqrt(1.0f - cos_theta * cos_theta),
          cos_phi = cosf(phi),
          sin_phi = sinf(phi);
    dir = Vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    pdf = kPiInv * cos_theta;
}

inline QUALIFIER_DEVICE float PdfHemisCos(const Vec3 &dir_local)
{
    return kPiInv * dir_local.z;
}

inline QUALIFIER_DEVICE void SampleGgx(float x_1, float x_2, float roughness, Vec3 &dir, float &pdf)
{
    const float phi = 2.0f * kPi * x_2,
                cos_phi = cosf(phi),
                sin_phi = sinf(phi),
                alpha_2 = roughness * roughness;
    const float tan_theta_2 = alpha_2 * x_1 / (1.0f - x_1),
                cos_theta = 1.0f / sqrt(1.0f + tan_theta_2),
                sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    dir = Vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    pdf = 1.0 / (kPi * alpha_2 * pow(cos_theta, 3) * pow(1.0f + tan_theta_2 / alpha_2, 2));
}

inline QUALIFIER_DEVICE float PdfGgx(float roughness, const Vec3 &normal, const Vec3 &h)
{
    const float cos_theta = Dot(normal, h);
    if (cos_theta <= 0.0f)
        return 0.0f;
    const float cos_theta_2 = cos_theta * cos_theta,
                tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2,
                cos_theta_3 = pow(cos_theta, 3),
                alpha_2 = roughness * roughness;
    return alpha_2 / (kPi * cos_theta_3 * pow(alpha_2 + tan_theta_2, 2));
}

inline QUALIFIER_DEVICE float SmithG1Ggx(float roughness, const Vec3 &v, const Vec3 &normal, const Vec3 &h)
{
    const float cos_theta = Dot(v, normal);
    if (cos_theta * Dot(v, h) <= 0.0f)
        return 0.0f;
    const float cos_theta_2 = cos_theta * cos_theta,
                tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2,
                alpha_2 = roughness * roughness;
    return 2.0f / (1.0f + sqrt(1.0f + alpha_2 * tan_theta_2));
}

inline QUALIFIER_DEVICE float MisWeight(float pdf1, float pdf2)
{
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return pdf1 / (pdf1 + pdf2);
}
