#pragma once

#include "thin_dielectric.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 薄的电介质材质
ThinDielectric::ThinDielectric(Float int_ior,
                               Float ext_ior,
                               std::unique_ptr<Texture> specular_reflectance,
                               std::unique_ptr<Texture> specular_transmittance)
    : Material(MaterialType::kThinDielectric),
      eta_inv_(ext_ior / int_ior),
      specular_reflectance_(std::move(specular_reflectance)),
      specular_transmittance_(std::move(specular_transmittance)) {}

///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
void ThinDielectric::Sample(BsdfSampling &bs) const
{
    auto kr = Fresnel(-bs.wo, bs.normal, eta_inv_);

    //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
    if (kr < 1)
        kr *= 2.0 / (1.0 + kr);

    if (UniformFloat() < kr)
    {
        bs.pdf = kr;
        bs.wi = -Reflect(-bs.wo, bs.normal);
        bs.attenuation = Vector3(kr);
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

    if (bs.pdf < kEpsilon)
    {
        bs.pdf = 0;
        return;
    }
}

///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
Spectrum ThinDielectric::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    auto kr = Fresnel(wi, normal, eta_inv_);
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
        auto attenuation = Vector3(1.0 - kr);
        if (specular_reflectance_)
            attenuation *= specular_reflectance_->Color(texcoord);
        return attenuation;
    }
    else
        return Spectrum(0);
}

///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
Float ThinDielectric::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    auto kr = Fresnel(wi, normal, eta_inv_);
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
bool ThinDielectric::TextureMapping() const
{
    return Material::TextureMapping() ||
           specular_reflectance_ && !specular_reflectance_->Constant() ||
           specular_transmittance_ && !specular_transmittance_->Constant();
}

NAMESPACE_END(simple_renderer)