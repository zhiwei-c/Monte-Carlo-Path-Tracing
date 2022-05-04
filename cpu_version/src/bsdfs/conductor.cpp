#include "conductor.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 平滑的导体材质
Conductor::Conductor(bool mirror,
                     const Spectrum &eta,
                     const Spectrum &k,
                     Float ext_ior,
                     std::unique_ptr<Texture> specular_reflectance)
    : Material(MaterialType::kConductor),
      mirror_(mirror),
      eta_(eta / ext_ior),
      k_(k / ext_ior),
      specular_reflectance_(std::move(specular_reflectance))
{
}

///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
void Conductor::Sample(BsdfSampling &bs) const
{
    bs.wi = -Reflect(-bs.wo, bs.normal);
    bs.pdf = 1;

    if (!bs.get_attenuation)
        return;

    bs.attenuation = mirror_ ? Spectrum(1) : FresnelConductor(bs.wi, bs.normal, eta_, k_);
    if (specular_reflectance_)
        bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
}

///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
Spectrum Conductor::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    if (!SameDirection(wo, Reflect(wi, normal)))
        return Spectrum(0);
    auto albedo = mirror_ ? Spectrum(1) : FresnelConductor(wi, normal, eta_, k_);
    if (specular_reflectance_)
        albedo *= specular_reflectance_->Color(texcoord);
    return albedo;
}

///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
Float Conductor::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    return SameDirection(wo, Reflect(wi, normal)) ? 1 : 0;
}

///\brief 是否映射纹理
bool Conductor::TextureMapping() const
{
    return Material::TextureMapping() ||
           specular_reflectance_ && !specular_reflectance_->Constant();
}

NAMESPACE_END(simple_renderer)