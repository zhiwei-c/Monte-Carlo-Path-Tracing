#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 薄的电介质材质派生类
class ThinDielectric : public Material
{
public:
    ///\brief 薄的电介质材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ///\param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ThinDielectric(Float int_ior, Float ext_ior,
                   std::unique_ptr<Texture> specular_reflectance,
                   std::unique_ptr<Texture> specular_transmittance)
        : Material(MaterialType::kThinDielectric),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)),
          specular_transmittance_(std::move(specular_transmittance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        Float kr = Fresnel(-bs.wo, bs.normal, eta_inv_);
        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2.0 / (1.0 + kr);
        if (UniformFloat() < kr)
        {
            bs.pdf = kr;
            bs.wi = -Reflect(-bs.wo, bs.normal);
            bs.attenuation = Spectrum(kr);
            if (specular_reflectance_)
                bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
        }
        else
        {
            bs.pdf = 1.0 - kr;
            bs.wi = bs.wo;
            bs.attenuation = Vector3(1 - kr);
            if (specular_transmittance_)
                bs.attenuation *= specular_transmittance_->Color(bs.texcoord);
        }
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
                  bool inside) const override
    {
        Float kr = Fresnel(wi, normal, eta_inv_);
        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2.0 / (1.0 + kr);
        if (SameDirection(wo, Reflect(wi, normal)))
        {
            auto attenuation = Spectrum(kr);
            if (specular_reflectance_)
                attenuation *= specular_reflectance_->Color(texcoord);
            return attenuation;
        }
        else if (SameDirection(wo, wi))
        {
            auto attenuation = Spectrum(1.0 - kr);
            if (specular_reflectance_)
                attenuation *= specular_reflectance_->Color(texcoord);
            return attenuation;
        }
        else
            return Spectrum(0);
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
              bool inside) const override
    {
        Float kr = Fresnel(wi, normal, eta_inv_);
        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2.0 / (1.0 + kr);
        if (SameDirection(wo, Reflect(wi, normal)))
            return kr;
        else if (SameDirection(wo, wi))
            return 1 - kr;
        else
            return 0;
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               specular_reflectance_ && !specular_reflectance_->Constant() ||
               specular_transmittance_ && !specular_transmittance_->Constant();
    }

private:
    Float eta_inv_;                                   //光线射出材质的相对折射率
    std::unique_ptr<Texture> specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    std::unique_ptr<Texture> specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)