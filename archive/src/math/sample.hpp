#pragma once

#include <random>

#include "coordinate.hpp"
#include "math.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

// Tiny Encryption Algorithm (TEA) to calculate a the sampler per launch index and iteration.
inline unsigned int Tea(const unsigned int val0, const unsigned int val1, const unsigned int N)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

///\brief 已知样本容量，生成前两维的哈默斯利序列
inline dvec2 Hammersley(uint32_t i, uint32_t N)
{
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    auto rdi = static_cast<double>(double(bits) * 2.3283064365386963e-10);
    return {double(i) / double(N), rdi};
}

///\brief 均匀抽样球面
inline dvec3 SampleSphereUniform(const dvec2 &sample)
{
    const double cos_theta = 1.0 - 2.0 * sample.x,
                 phi = 2.0 * kPi * sample.y,
                 sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    return dvec3{sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
}

///\brief 余弦加权抽样单位半球，样本方向在给定竖直向上方向的局部坐标系中指向原点
inline double SampleHemisCos(const dvec3 &up, dvec3 *dir, const dvec2 &sample)
{
    double cos_theta = std::sqrt(sample.x),
           phi = 2.0 * kPi * sample.y;
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    *dir = -ToWorld(dvec3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta), up);
    return kPiRcp * cos_theta;
}

///\brief 均匀抽样锥体
inline dvec3 SampleConeUniform(double cos_cutoff, const dvec2 &sample)
{
    double cos_theta = (1 - sample.x) + sample.x * cos_cutoff;
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    double phi = 2.0 * kPi * sample.y;
    return dvec3(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
}

///\brief 计算余弦加权抽样单位半球的概率
inline double PdfHemisCos(const dvec3 &dir, const dvec3 &up)
{
    double cos_theta = ToLocal(dir, up).z;
    return kPiRcp * cos_theta;
}

///\brief 计算多重重要性抽样的权重
inline double MisWeight(double pdf1, double pdf2)
{
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return pdf1 / (pdf1 + pdf2);
}

///\brief 加权平均多重重要性抽样生成的样本结果
inline dvec3 WeightPowerHeuristic(const std::vector<dvec3> &values, std::vector<double> pdfs)
{
    double weight_sum = 0;
    for (double &pdf : pdfs)
    {
        pdf *= pdf;
        weight_sum += pdf;
    }

    auto result = dvec3(0);
    for (int i = 0; i < values.size(); ++i)
    {
        result += values[i] * (pdfs[i] / weight_sum);
    }

    return result;
}

inline size_t SampleCdf(const std::vector<double> &cdf, size_t size, double sample)
{
    auto position = std::lower_bound(cdf.begin(), cdf.end(), sample);
    return std::min(std::max(static_cast<size_t>(position - cdf.begin() - 1),
                             static_cast<size_t>(0)),
                    static_cast<size_t>(size - 1));
}

NAMESPACE_END(raytracer)