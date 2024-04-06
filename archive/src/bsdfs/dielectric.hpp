#pragma once

#include "bsdf.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//平滑的电介质
class Dielectric : public Bsdf
{
public:
    Dielectric(const std::string &id, double int_ior, double ext_ior, Texture *specular_reflectance,
               Texture *specular_transmittance);

    void Sample(SamplingRecord *rec, Sampler* sampler) const override;
    void Eval(SamplingRecord *rec) const override;

private:
    bool UseTextureMapping() const override;

    double eta_;                      //介质折射率与外部折射率之比
    double eta_inv_;                  //外部折射率与介质折射率之比
    Texture *specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1）
    Texture *specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1）
};
NAMESPACE_END(raytracer)