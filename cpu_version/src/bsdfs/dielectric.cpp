#include "dielectric.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 平滑的电介质材质
Dielectric::Dielectric(Float int_ior,
                       Float ext_ior,
                       std::unique_ptr<Texture> specular_reflectance,
                       std::unique_ptr<Texture> specular_transmittance)
    : Material(MaterialType::kDielectric),
      eta_(int_ior / ext_ior),
      eta_inv_(ext_ior / int_ior),
      specular_reflectance_(std::move(specular_reflectance)),
      specular_transmittance_(std::move(specular_transmittance)) {}

///\brief 根据光线出射方向和表面法线方向抽样光线入射方向，法线方向已被处理至与光线出射方向夹角大于90度
void Dielectric::Sample(BsdfSampling &bs) const
{
    auto eta = bs.inside ? eta_inv_ : eta_;     //相对折射率，即光线透射侧介质折射率与入透射侧介质折射率之比
    auto eta_inv = bs.inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

    auto kr = Fresnel(-bs.wo, bs.normal, eta_inv);
    if (UniformFloat() < kr)
    {
        bs.pdf = kr;
        bs.wi = -Reflect(-bs.wo, bs.normal);
        if (bs.get_attenuation)
        {
            bs.attenuation = Spectrum(kr);
            if (specular_reflectance_)
                bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
        }
    }
    else
    {
        bs.wi = -Refract(-bs.wo, bs.normal, eta_inv);
        auto kr_t = Fresnel(bs.wi, -bs.normal, eta);
        bs.pdf = 1 - kr_t;
        if (bs.get_attenuation)
        {
            bs.attenuation = Spectrum(1 - kr_t);
            if (specular_transmittance_)
                bs.attenuation *= specular_transmittance_->Color(bs.texcoord);
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            bs.attenuation *= Sqr(eta);
        }
    }

    if (bs.pdf < kEpsilonL)
        bs.pdf = 0;
}

///\brief 根据光线入射方向、出射方向和表面法线方向，计算 BSDF 权重，法线方向已被处理至与光线入射方向夹角大于90度
Spectrum Dielectric::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
    auto kr = Fresnel(wi, normal, eta_inv);
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

///\brief 根据光线入射方向和表面法线方向，计算光线从给定出射方向射出的概率，法线方向已被处理至与光线入射方向夹角大于90度
Float Dielectric::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
    auto kr = Fresnel(wi, normal, eta_inv);
    if (SameDirection(wo, Reflect(wi, normal)))
        return kr;
    else if (kr > kOneMinusEpsilon)
        return 0;
    else if (SameDirection(wo, Refract(wi, normal, eta_inv)))
        return 1 - kr;
    else
        return 0;
}

///\brief 是否映射纹理
bool Dielectric::TextureMapping() const
{
    return Material::TextureMapping() ||
           specular_reflectance_ && !specular_reflectance_->Constant() ||
           specular_transmittance_ && !specular_transmittance_->Constant();
}

NAMESPACE_END(simple_renderer)