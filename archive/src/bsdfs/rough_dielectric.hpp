#pragma once

#include "bsdf.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//粗糙的电介质
class RoughDielectric : public Bsdf
{
public:
    RoughDielectric(const std::string &id, double int_ior, double ext_ior, Ndf *ndf, Texture *specular_reflectance,
                    Texture *specular_transmittance);
    ~RoughDielectric();

    void Sample(SamplingRecord *rec, Sampler *sampler) const override;
    void Eval(SamplingRecord *rec) const override;

private:
    double EvalMultipleScatter(double cos_theta_i, double cos_theta_o, bool inside) const;
    bool UseTextureMapping() const override;

    Ndf *ndf_;                        //微表面法线分布
    double f_add_;                    //入射光线在物体外部时，补偿多次散射后出射光能的系数
    double f_add_inv_;                //入射光线在物体内部时，补偿多次散射后出射光能的系数
    double ratio_t_;                  //入射光线在物体外部时，补偿多次散射后出射光能中折射的比例
    double ratio_t_inv_;              //入射光线在物体内部时，补偿多次散射后出射光能中折射的比例
    double eta_;                      //介质折射率与外部折射率之比
    double eta_inv_;                  //外部折射率与介质折射率之比
    Texture *specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1）
    Texture *specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1）
};
NAMESPACE_END(raytracer)