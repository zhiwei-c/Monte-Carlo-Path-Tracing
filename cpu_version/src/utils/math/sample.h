#pragma once

#include <utility>
#include <random>

#include "coordinate.h"

NAMESPACE_BEGIN(raytracer)

///\brief 获取一个在 [0,1] 之间的随机数
inline Float UniformFloat()
{
    std::random_device dev;
    std::default_random_engine e(dev());
    std::uniform_real_distribution<Float> dist;
    return std::abs(1.0 - dist(e) * 2.0);
}

///\brief 获取一个在 [0,1) 之间的随机数
inline Float UniformFloat2()
{
    std::random_device dev;
    std::default_random_engine e(dev());
    std::uniform_real_distribution<Float> dist;
    return dist(e);
}

///\brief 已知样本容量，生成前两维的哈默斯利序列
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

///\brief 均匀抽样单位球
inline Vector3 SphereUniform()
{
    Float x_1 = UniformFloat(), x_2 = UniformFloat();
    Float cos_theta = 1.0 - 2.0 * x_2, phi = 2.0 * kPi * x_1;
    auto sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    return Vector3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
}

///\brief 均匀抽样半径 1 的圆盘
inline Vector2 DiskUnifrom()
{
    Float r1 = 2.0 * UniformFloat() - 1.0, r2 = 2.0 * UniformFloat() - 1.0;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
    Float phi = 0, radius = 0;
    if (r1 == 0 && r2 == 0)
        radius = phi = 0;
    else if (r1 * r1 > r2 * r2)
    {
        radius = r1;
        phi = (kPi / 4) * (r2 / r1);
    }
    else
    {
        radius = r2;
        phi = (kPi / 2) - (r1 / r2) * (kPi / 4);
    }
    return Vector2(radius * std::cos(phi), radius * std::sin(phi));
}

///\brief 均匀抽样单位半球，得到的方向指向原点
inline void SampleHemisUniform(const Vector3 &up, Vector3 &dir, Float &pdf)
{
    Float x_1 = UniformFloat(), x_2 = UniformFloat();
    Float cos_theta = x_1, phi = 2.0 * kPi * x_2;
    Float sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    dir = -ToWorld(Vector3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta), up);
    pdf = kPiInv * 0.5;
}

///\brief 余弦加权抽样单位半球，得到的方向指向原点
inline void SampleHemisCos(const Vector3 &up, Vector3 &dir, Float *pdf = nullptr)
{
    Float x_1 = UniformFloat(), x_2 = UniformFloat();
    Float cos_theta = std::sqrt(x_1), phi = 2.0 * kPi * x_2;
    Float sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    dir = -ToWorld(Vector3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta), up);
    if(pdf)
        *pdf = kPiInv * cos_theta;
}

///\brief 计算余弦加权抽样单位半球的概率
inline Float PdfHemisCos(const Vector3 &dir, const Vector3 &up)
{
    Float cos_theta = ToLocal(dir, up).z;
    return kPiInv * cos_theta;
}

///\brief n 次方余弦加权抽样单位半球，得到的方向指向原点
inline Vector3 SampleHemisCosN(const Float n, const Vector3 &up)
{
    Float x_1 = UniformFloat(), x_2 = UniformFloat();
    Float cos_theta = std::pow(x_1, 1.0 / (n + 1)), phi = 2.0 * kPi * x_2;
    Float sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    return -ToWorld(Vector3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta), up);
}

///\brief n 次方余弦加权抽样单位半球的概率
inline Float PdfHemisCosN(const Vector3 &dir, const Vector3 &up, const Float n)
{
    Float cos_theta = ToLocal(dir, up).z;
    return (n + 1) * 0.5 * kPiInv * std::pow(cos_theta, n);
}

///\brief 计算多重重要性抽样的权重
inline Float MisWeight(Float pdf1, Float pdf2)
{
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return pdf1 / (pdf1 + pdf2);
}

///\brief 加权平均多重重要性抽样生成的样本结果
inline Spectrum WeightPowerHeuristic(const std::vector<Spectrum> &values, std::vector<Float> pdfs)
{
    Float weight_sum = 0;
    for (auto &pdf : pdfs)
    {
        pdf *= pdf;
        weight_sum += pdf;
    }

    auto result = Spectrum(0);
    for (int i = 0; i < values.size(); i++)
        result += values[i] * (pdfs[i] / weight_sum);

    return result;
}

NAMESPACE_END(raytracer)