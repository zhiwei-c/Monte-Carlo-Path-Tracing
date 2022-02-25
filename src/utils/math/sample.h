#pragma once

#include <utility>
#include <random>

#include "math.h"

NAMESPACE_BEGIN(simple_renderer)

//\brief 获取一个在 [0,1] 之间的随机数
//
//\return 得到的随机数
inline Float UniformFloat()
{
    std::random_device dev;
    std::default_random_engine e(dev());
    std::uniform_real_distribution<Float> dist;

    return std::fabs(1 - dist(e) * 2);
}

//\brief 获取一个在 [0,1) 之间的随机数
//
//\return 得到的随机数
inline Float UniformFloat2()
{
    std::random_device dev;
    std::default_random_engine e(dev());
    std::uniform_real_distribution<Float> dist;
    return dist(e);
}

inline Vector2 Hammersley(uint32_t i, uint32_t N)
{
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    Float rdi = Float(bits) * 2.3283064365386963e-10;
    return {Float(i) / Float(N), rdi};
}

inline Vector3 SphereUniform()
{
    auto x_1 = UniformFloat();
    auto x_2 = UniformFloat();

    auto cos_theta = 1 - 2 * x_2;
    auto sin_theta = std::sqrt(std::max((decltype(cos_theta))0, 1 - cos_theta * cos_theta));

    auto phi = 2 * kPi * x_1;
    auto cos_phi = std::cos(phi),
         sin_phi = std::sin(phi);

    return Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

inline Vector2 DiskUnifrom()
{

    auto x_1 = UniformFloat();
    auto x_2 = UniformFloat();
    Float r1 = 2. * x_1 - 1;
    Float r2 = 2. * x_2 - 1;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
    Float phi, r;
    if (r1 == 0 && r2 == 0)
    {
        r = phi = 0;
    }
    else if (r1 * r1 > r2 * r2)
    {
        r = r1;
        phi = (kPi / 4) * (r2 / r1);
    }
    else
    {
        r = r2;
        phi = (kPi / 2) - (r1 / r2) * (kPi / 4);
    }
    auto cos_phi = std::cos(phi),
         sin_phi = std::sin(phi);

    return Vector2(r * cos_phi, r * sin_phi);
}

inline std::pair<Vector3, Float> HemisUniform()
{
    auto x_1 = UniformFloat();
    auto x_2 = UniformFloat();
    auto cos_theta = x_1,
         phi = 2.f * kPi * x_2;
    auto sin_theta = std::sqrt(1.f - cos_theta * cos_theta),
         cos_phi = std::cos(phi),
         sin_phi = std::sin(phi);
    auto dir = Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    auto pdf = kPiInv * 0.5;
    return {dir, pdf};
}

inline std::pair<Vector3, Float> HemisCos()
{
    auto x_1 = UniformFloat();
    auto x_2 = UniformFloat();
    auto cos_theta = std::sqrt(x_1),
         phi = 2.f * kPi * x_2;
    auto sin_theta = std::sqrt(1.f - cos_theta * cos_theta),
         cos_phi = std::cos(phi),
         sin_phi = std::sin(phi);
    auto dir = Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    auto pdf = kPiInv * cos_theta;
    return {dir, pdf};
}

inline Float PdfHemisCos(const Vector3 &dir_local)
{
    auto cos_theta = dir_local.z;
    auto pdf = kPiInv * cos_theta;
    return pdf;
}

inline Vector3 HemisCosN(const Float n)
{
    auto x_1 = UniformFloat();
    auto x_2 = UniformFloat();
    auto cos_theta = std::pow(x_1, 1.f / (n + 1)),
         phi = 2.f * kPi * x_2;
    auto sin_theta = std::sqrt(1.f - cos_theta * cos_theta),
         cos_phi = std::cos(phi),
         sin_phi = std::sin(phi);
    auto dir = Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    return dir;
}

inline Float PdfHemisCosN(const Vector3 &dir_local, const Float n)
{
    auto cos_theta = dir_local.z;
    auto pdf = (n + 1) * 0.5 * kPiInv * std::pow(cos_theta, n);
    return pdf;
}

NAMESPACE_END(simple_renderer)