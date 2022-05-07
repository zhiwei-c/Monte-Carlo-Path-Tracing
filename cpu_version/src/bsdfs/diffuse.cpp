#include "diffuse.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 平滑的理想漫反射材质
Diffuse::Diffuse(std::unique_ptr<Texture> reflectance)
    : Material(MaterialType::kDiffuse),
      reflectance_(std::move(reflectance)) {}

///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
void Diffuse::Sample(BsdfSampling &bs) const
{
    auto [wi_local, pdf] = HemisCos();
    if (pdf < kEpsilonL)
        return;

    bs.wi = -ToWorld(wi_local, bs.normal);
    bs.pdf = pdf;

    if (!bs.get_attenuation)
        return;

    bs.attenuation = reflectance_->Color(bs.texcoord) * kPiInv;
}

///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
Spectrum Diffuse::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    return reflectance_->Color(texcoord) * kPiInv;
}

///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
Float Diffuse::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    // 入射、出射光线需在同侧
    if (NotSameHemis(wo, -wi))
        return 0;
        
    auto wo_local = ToLocal(wo, normal);
    return PdfHemisCos(wo_local);
}

///\brief 是否映射纹理
bool Diffuse::TextureMapping() const
{
    return Material::TextureMapping() ||
           !reflectance_->Constant();
}

///\brief 给定点是否透明
bool Diffuse::Transparent(const Vector2 &texcoord) const
{
    return Material::Transparent(texcoord) ||
           reflectance_->Transparent(texcoord);
}

NAMESPACE_END(simple_renderer)