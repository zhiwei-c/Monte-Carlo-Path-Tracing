#include "glossy.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 冯模型描述的有光泽材质
Glossy::Glossy(std::unique_ptr<Texture> diffuse_reflectance,
               std::unique_ptr<Texture> specular_reflectance,
               Float exponent)
    : Material(MaterialType::kGlossy),
      diffuse_reflectance_(std::move(diffuse_reflectance)),
      specular_reflectance_(std::move(specular_reflectance)),
      exponent_(exponent)
{
    diffuse_reflectance_sum_ = -1,
    specular_reflectance_sum_ = -1;
    diffuse_sampling_weight_ = -1;
    if (diffuse_reflectance_->Constant())
    {
        auto kd = diffuse_reflectance->Color(Vector2(0));
        diffuse_reflectance_sum_ = kd.r + kd.g + kd.b;
    }
    if (specular_reflectance->Constant())
    {
        auto ks = specular_reflectance->Color(Vector2(0));
        specular_reflectance_sum_ = ks.r + ks.g + ks.b;
    }
    if (diffuse_reflectance_->Constant() && specular_reflectance->Constant())
        diffuse_sampling_weight_ = diffuse_reflectance_sum_ / (diffuse_reflectance_sum_ + specular_reflectance_sum_);
}

///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
void Glossy::Sample(BsdfSampling &bs) const
{
    auto pdf_diffuse = DiffuseSamplingWeight(bs.texcoord);
    auto sample_x = UniformFloat();
    if (sample_x < pdf_diffuse)
    {
        auto [wi_local, pdf] = HemisCos();
        bs.wi = -ToWorld(wi_local, bs.normal);
    }
    else
    {
        auto wi_local = HemisCosN(exponent_);
        auto wr_pseudo = Reflect(-bs.wo, bs.normal);
        auto wi = -ToWorld(wi_local, wr_pseudo);
        if (SameHemis(wi, bs.normal))
            return;
    }
    bs.pdf = Pdf(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    if (bs.pdf < kEpsilonL)
    {
        bs.pdf = 0;
        return;
    }

    if (bs.get_attenuation)
        bs.attenuation = Eval(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
}

///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
Spectrum Glossy::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    // 入射、出射光线需在同侧
    if (NotSameHemis(wo, normal))
        return Spectrum(0);

    Spectrum albedo(0);
    // 计算漫反射分量的贡献
    albedo += diffuse_reflectance_->Color(texcoord) * kPiInv;
    // 计算镜面反射分量的贡献
    auto wr = Reflect(wi, normal);
    auto cos_alpha = glm::dot(wr, wo);
    if (cos_alpha > kEpsilon)
        albedo += specular_reflectance_->Color(texcoord) *
                  static_cast<Float>((exponent_ + 2) * kPiInv * 0.5 * std::pow(cos_alpha, exponent_));

    return albedo;
}

///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
Float Glossy::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    if (NotSameHemis(wo, normal))
        return 0;

    auto pdf_diffuse = DiffuseSamplingWeight(texcoord);
    auto wo_local = ToLocal(wo, normal);
    auto pdf = pdf_diffuse * PdfHemisCos(wo_local);

    auto wr = Reflect(wi, normal);
    if (SameHemis(wo, wr))
    {
        wo_local = ToLocal(wo, wr);
        pdf += (1 - pdf_diffuse) * PdfHemisCosN(wo_local, exponent_);
    }
    return pdf;
}

///\brief 是否映射纹理
bool Glossy::TextureMapping() const
{
    return Material::TextureMapping() ||
           !diffuse_reflectance_->Constant() ||
           !specular_reflectance_->Constant();
}

///\brief 给定点是否透明
bool Glossy::Transparent(const Vector2 &texcoord) const
{
    return Material::Transparent(texcoord) ||
           diffuse_reflectance_->Transparent(texcoord);
}

///\brief 获取给定点抽样漫反射的权重
Float Glossy::DiffuseSamplingWeight(const Vector2 &texcoord) const
{
    if (diffuse_sampling_weight_ >= 0)
        return diffuse_sampling_weight_;

    auto ks_sum = specular_reflectance_sum_;
    if (ks_sum < 0)
    {
        auto ks = specular_reflectance_->Color(texcoord);
        ks_sum = ks.r + ks.g + ks.b;
    }

    auto kd_sum = diffuse_reflectance_sum_;
    if (kd_sum < 0)
    {
        auto kd = diffuse_reflectance_->Color(texcoord);
        kd_sum = kd.r + kd.g + kd.b;
    }

    return kd_sum / (kd_sum + ks_sum);
}

NAMESPACE_END(simple_renderer)