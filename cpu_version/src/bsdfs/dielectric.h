#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 平滑的电介质派生类
class Dielectric : public Material
{
public:
    ///\brief 平滑的电介质材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ///\param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    Dielectric(Float int_ior, Float ext_ior,
               std::unique_ptr<Texture> specular_reflectance,
               std::unique_ptr<Texture> specular_transmittance)
        : Material(MaterialType::kDielectric),
          eta_(int_ior / ext_ior), eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)),
          specular_transmittance_(std::move(specular_transmittance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        Float eta = bs.inside ? eta_inv_ : eta_,      //相对折射率，即光线透射侧介质折射率与入透射侧介质折射率之比
            eta_inv = bs.inside ? eta_ : eta_inv_,    //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            kr = Fresnel(-bs.wo, bs.normal, eta_inv); //菲涅尔项
        if (UniformFloat() < kr)
        { //抽样反射光线
            bs.wi = -Reflect(-bs.wo, bs.normal);
            bs.pdf = kr;
            if (!bs.get_attenuation)
                return;
            bs.attenuation = Spectrum(kr);
            if (specular_reflectance_)
                bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
        }
        else
        { //抽样折射光线
            bs.wi = -Refract(-bs.wo, bs.normal, eta_inv);
            kr = Fresnel(bs.wi, -bs.normal, eta);
            bs.pdf = 1 - kr;
            if (!bs.get_attenuation)
                return;
            bs.attenuation = Spectrum(1.0 - kr);
            if (specular_transmittance_)
                bs.attenuation *= specular_transmittance_->Color(bs.texcoord);
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            bs.attenuation *= Sqr(eta);
        }
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
                  bool inside) const override
    {
        Float eta_inv = inside ? eta_ : eta_inv_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            kr = Fresnel(wi, normal, eta_inv);
        if (SameDirection(wo, Reflect(wi, normal)))
        {
            auto albedo = Spectrum(kr);
            if (specular_reflectance_)
                albedo *= specular_reflectance_->Color(texcoord);
            return albedo;
        }
        else if (SameDirection(wo, Refract(wi, normal, eta_inv)))
        {
            auto attenuation = Spectrum(1 - kr);
            if (specular_transmittance_)
                attenuation *= specular_transmittance_->Color(texcoord);
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            attenuation *= Sqr(eta_inv);
            return attenuation;
        }
        else
            return Spectrum(0);
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
              bool inside) const override
    {
        Float eta_inv = inside ? eta_ : eta_inv_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            kr = Fresnel(wi, normal, eta_inv);
        if (SameDirection(wo, Reflect(wi, normal)))
            return kr;
        else if (SameDirection(wo, Refract(wi, normal, eta_inv)))
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
    Float eta_;                                       //介质折射率与外部折射率之比
    Float eta_inv_;                                   //外部折射率与介质折射率之比
    std::unique_ptr<Texture> specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    std::unique_ptr<Texture> specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)