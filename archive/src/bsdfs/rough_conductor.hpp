#pragma once

#include "bsdf.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//粗糙的导体
class RoughConductor : public Bsdf
{
public:
    RoughConductor(const std::string &id, const dvec3 &eta, const dvec3 &k, Ndf *ndf, Texture *specular_reflectance);
    ~RoughConductor();

    void Sample(SamplingRecord *rec, Sampler *sampler) const override;
    void Eval(SamplingRecord *rec) const override;

private:
    dvec3 EvalMultipleScatter(double cos_theta_i, double cos_theta_o) const;
    bool UseTextureMapping() const override;

    Ndf *ndf_;                      //微表面法线分布
    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1）
    dvec3 eta_;                     //材质相对折射率的实部
    dvec3 k_;                       //材质相对折射率的虚部（消光系数）
    dvec3 f_add_;                   //补偿多次散射后出射光能的系数
};
NAMESPACE_END(raytracer)