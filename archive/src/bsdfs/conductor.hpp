#pragma once

#include "bsdf.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//平滑的导体
class Conductor : public Bsdf
{
public:
    Conductor(const std::string &id, const dvec3 &eta, const dvec3 &k, Texture *specular_reflectance);

    void Sample(SamplingRecord *rec, Sampler* sampler) const override;
    void Eval(SamplingRecord *rec) const override;

private:
    bool UseTextureMapping() const override;

    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1）
    dvec3 eta_;                     //材质相对折射率的实部
    dvec3 k_;                       //材质相对折射率的虚部（消光系数）
};
NAMESPACE_END(raytracer)