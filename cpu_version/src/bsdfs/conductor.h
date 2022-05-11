#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 平滑的导体材质派生类
class Conductor : public Material
{
public:
    ///\brief 平滑的导体材质
    ///\param eta 材质折射率的实部
    ///\param k 材质折射率的虚部（消光系数）
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    Conductor(const Spectrum &eta, const Spectrum &k, Float ext_ior,
              std::unique_ptr<Texture> specular_reflectance)
        : Material(MaterialType::kConductor),
          eta_(eta / ext_ior), k_(k / ext_ior),
          specular_reflectance_(std::move(specular_reflectance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        bs.wi = -Reflect(-bs.wo, bs.normal);
        bs.pdf = 1;
        if (!bs.get_attenuation)
            return;
        bs.attenuation = FresnelConductor(bs.wi, bs.normal, eta_, k_);
        if (specular_reflectance_)
            bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
                  bool inside) const override
    {
        if (!SameDirection(wo, Reflect(wi, normal)))
            return Spectrum(0);
        Spectrum albedo = FresnelConductor(wi, normal, eta_, k_);
        if (specular_reflectance_)
            albedo *= specular_reflectance_->Color(texcoord);
        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
              bool inside) const override
    {
        return SameDirection(wo, Reflect(wi, normal)) ? 1 : 0;
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               specular_reflectance_ && !specular_reflectance_->Constant();
    }

private:
    Spectrum eta_;                                  //材质相对折射率的实部
    Spectrum k_;                                    //材质相对折射率的虚部（消光系数）
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)