#pragma once

#include "bsdf.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

// 涂上一层清漆（电介质）的导体
class ClearCoatedConductor : public Bsdf
{
public:
    ClearCoatedConductor(const std::string &id, const dvec3 &eta, const dvec3 &k, Ndf *ndf, double clear_coat, Ndf *ndf_coat, Texture *specular_reflectance);
    ~ClearCoatedConductor();

    void Sample(SamplingRecord *rec, Sampler *sampler) const override;
    void Eval(SamplingRecord *rec) const override;

private:
    Ndf *ndf_coat_;          // 微表面法线分布
    Bsdf *nested_conductor_; // 基底
    double clear_coat_;      // 清漆强度
};
NAMESPACE_END(raytracer)