#pragma once

#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 薄的电介质材质派生类
class ThinDielectric : public Bsdf
{
public:
    ///\brief 薄的电介质材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ///\param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ThinDielectric(Float int_ior, Float ext_ior, std::unique_ptr<Texture> specular_reflectance,
                   std::unique_ptr<Texture> specular_transmittance)
        : Bsdf(BsdfType::kThinDielectric), eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)), specular_transmittance_(std::move(specular_transmittance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(SamplingRecord &rec) const override
    {
        Float kr = Fresnel(-rec.wo, rec.normal, eta_inv_);
        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2.0 / (1.0 + kr);
        if (UniformFloat() < kr)
        { //抽样反射光线
            rec.pdf = kr;
            if (rec.pdf < kEpsilonPdf)
                return;
            rec.wi = -Reflect(-rec.wo, rec.normal);
            rec.type = ScatteringType::kReflect;
            rec.attenuation = Spectrum(kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        }
        else
        { //抽样折射光线
            rec.pdf = 1.0 - kr;
            if (rec.pdf < kEpsilonPdf)
                return;
            rec.wi = rec.wo;
            rec.type = ScatteringType::kTransimission;
            rec.attenuation = Vector3(1.0 - kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
        }
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    void Eval(SamplingRecord &rec) const override
    {
        Float kr = Fresnel(rec.wi, rec.normal, eta_inv_);
        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2.0 / (1.0 + kr);
        if (SameDirection(rec.wo, Reflect(rec.wi, rec.normal)))
        { //处理镜面反射
            rec.pdf = kr;
            rec.type = ScatteringType::kReflect;
            rec.attenuation = Spectrum(kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        }
        else if (SameDirection(rec.wo, rec.wi))
        { //处理折射
            rec.pdf = 1.0 - kr;
            rec.type = ScatteringType::kTransimission;
            rec.attenuation = Spectrum(1.0 - kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
        }
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || specular_reflectance_ && !specular_reflectance_->Constant() ||
               specular_transmittance_ && !specular_transmittance_->Constant();
    }

private:
    Float eta_inv_;                                   //光线射出材质的相对折射率
    std::unique_ptr<Texture> specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    std::unique_ptr<Texture> specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)