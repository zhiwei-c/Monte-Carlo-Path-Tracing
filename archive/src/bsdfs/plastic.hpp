#pragma once

#include "bsdf.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//平滑的塑料
class Plastic : public Bsdf
{
public:
    Plastic(const std::string &id, double int_ior, double ext_ior, Texture *diffuse_reflectance,
            Texture *specular_reflectance, bool nonlinear);

    void Sample(SamplingRecord *rec, Sampler* sampler) const override;
    void Eval(SamplingRecord *rec) const override;

private:
    bool UseTextureMapping() const override;
    double SpecularSamplingWeight(const dvec2 &texcoord) const;

    bool nonlinear_;                //是否考虑因内部散射而引起的非线性色移
    double fdr_;                    //漫反射菲涅尔项的平均值
    double eta_inv_;                //外部折射率与介质折射率之比
    Texture *diffuse_reflectance_;  //漫反射系数
    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};
NAMESPACE_END(raytracer)