#pragma once

#include "bsdf.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//平滑的理想漫反射材质，由朗伯模型描述
class Diffuse : public Bsdf
{
public:
    Diffuse(const std::string &id, Texture *reflectance);

    void Sample(SamplingRecord *rec, Sampler* sampler) const override;
    void Eval(SamplingRecord *rec) const override;

    bool IsTransparent(const dvec2 &texcoord, Sampler* sampler) const override;

private:
    bool UseTextureMapping() const override;

    Texture *reflectance_; //漫反射系数
};
NAMESPACE_END(raytracer)