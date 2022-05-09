#include "rough_plastic.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 粗糙的塑料材质
RoughPlastic::RoughPlastic(Float int_ior,
                           Float ext_ior,
                           std::unique_ptr<Texture> diffuse_reflectance,
                           std::unique_ptr<Texture> specular_reflectance,
                           MicrofacetDistribType distrib_type,
                           std::unique_ptr<Texture> alpha,
                           bool nonlinear)
    : Material(MaterialType::kRoughPlastic),
      Microfacet(distrib_type,
                 std::move(alpha),
                 nullptr),
      eta_inv_(ext_ior / int_ior),
      fdr_(AverageFresnel(int_ior / ext_ior)),
      diffuse_reflectance_(std::move(diffuse_reflectance)),
      specular_reflectance_(std::move(specular_reflectance)),
      nonlinear_(nonlinear)
{
    specular_sampling_weight_ = -1;
    if (diffuse_reflectance_->Constant() &&
        (!specular_reflectance_ || specular_reflectance_->Constant()))
    {
        auto kd = diffuse_reflectance_->Color(Vector2(0));
        auto d_sum = kd.r + kd.g + kd.b;
        auto s_sum = 3.0;
        if (specular_reflectance_)
        {
            auto ks = specular_reflectance_->Color(Vector2(0));
            s_sum = ks.r + ks.g + ks.b;
        }
        specular_sampling_weight_ = s_sum / (d_sum + s_sum);
    }

    if (Material::TextureMapping())
    {
        albedo_avg_ = -1;
        return;
    }

    ComputeAlbedoTable();
    if (albedo_avg_ < 0)
        return;

    f_add_ = Sqr(fdr_) * albedo_avg_ / (1.0 - fdr_ * (1.0 - albedo_avg_));
}

///\brief 根据光线出射方向和表面法线方向抽样光线入射方向，法线方向已被处理至与光线出射方向夹角大于90度
void RoughPlastic::Sample(BsdfSampling &bs) const
{
    auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);
    auto specular_sampling_weight = SpecularSamplingWeight(bs.texcoord);

    auto kr_o = Fresnel(-bs.wo, bs.normal, eta_inv_);
    auto pdf_specular = kr_o * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

    auto h = Vector3(0);
    auto D = static_cast<Float>(0);
    auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_u);
    if (UniformFloat() < pdf_specular)
    {
        std::tie(h, D) = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});
        bs.wi = -Reflect(-bs.wo, h);
        if (glm::dot(bs.wi, bs.normal) >= 0)
            return;
    }
    else
    {
        auto [wi_local, pdf] = HemisCos();
        bs.wi = -ToWorld(wi_local, bs.normal);
        h = glm::normalize(-bs.wi + bs.wo);
        D = distrib->Pdf(h, bs.normal);
    }

    auto kr_i = Fresnel(bs.wi, bs.normal, eta_inv_);
    pdf_specular = kr_i * specular_sampling_weight,
    pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    pdf_diffuse = 1.0 - pdf_specular;

    bs.pdf = pdf_diffuse * PdfHemisCos(ToLocal(bs.wo, bs.normal));
    if (D > kEpsilon)
        bs.pdf += pdf_specular * D * std::abs(1.0 / (4.0 * glm::dot(bs.wo, h)));
    if (bs.pdf < kEpsilonL)
    {
        bs.pdf = 0;
        return;
    }

    if (!bs.get_attenuation)
        return;

    auto diffuse_reflectance = diffuse_reflectance_->Color(bs.texcoord);
    if (nonlinear_)
        bs.attenuation = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
    else
        bs.attenuation = diffuse_reflectance / (1.0 - fdr_);
    bs.attenuation *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
    if (D > kEpsilon)
    {
        auto cos_i_n = glm::dot(bs.wi, bs.normal);
        auto cos_o_n = glm::dot(bs.wo, bs.normal);
        auto G = distrib->SmithG1(-bs.wi, h, bs.normal) *
                 distrib->SmithG1(bs.wo, h, bs.normal);
        auto F = Fresnel(bs.wi, h, eta_inv_);
        auto attenuation = F * D * G / (4.0 * std::abs(cos_i_n * cos_o_n));
        if (albedo_avg_ > 0)
            attenuation += EvalMultipleScatter(cos_i_n, cos_o_n);
        auto specular_reflectance = specular_reflectance_ ? specular_reflectance_->Color(bs.texcoord) : Spectrum(1);
        bs.attenuation += specular_reflectance * attenuation;
    }
}

///\brief 根据光线入射方向、出射方向和表面法线方向，计算 BSDF 权重，法线方向已被处理至与光线入射方向夹角大于90度
Spectrum RoughPlastic::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    auto [alpha_u, alpha_v] = GetAlpha(texcoord);

    auto cos_i_n = glm::dot(wi, normal);
    auto cos_o_n = glm::dot(wo, normal);

    auto albedo = Spectrum(0);

    auto diffuse_reflectance = diffuse_reflectance_->Color(texcoord);
    if (nonlinear_)
        albedo = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
    else
        albedo = diffuse_reflectance / (1.0 - fdr_);

    auto kr_i = Fresnel(wi, normal, eta_inv_);
    auto kr_o = Fresnel(-wo, normal, eta_inv_);
    albedo *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;

    auto h = glm::normalize(-wi + wo);
    auto F = Fresnel(wi, h, eta_inv_);

    auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_u);
    auto D = distrib->Pdf(h, normal);
    if (D > kEpsilon)
    {
        auto G = distrib->SmithG1(-wi, h, normal) *
                 distrib->SmithG1(wo, h, normal);
        auto attenuation = F * D * G / (4.0 * std::abs(cos_i_n * cos_o_n));
        if (albedo_avg_ > 0)
            attenuation += EvalMultipleScatter(cos_i_n, cos_o_n);

        auto specular_reflectance = specular_reflectance_ ? specular_reflectance_->Color(texcoord) : Spectrum(1);
        albedo += specular_reflectance * attenuation;
    }

    return albedo;
}

///\brief 根据光线入射方向和表面法线方向，计算光线从给定出射方向射出的概率，法线方向已被处理至与光线入射方向夹角大于90度
Float RoughPlastic::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    // 表面法线方向，光线入射和出射需在介质同侧
    if (NotSameHemis(wo, normal))
        return 0;

    auto [alpha_u, alpha_v] = GetAlpha(texcoord);

    auto kr = Fresnel(wi, normal, eta_inv_);
    auto specular_sampling_weight = SpecularSamplingWeight(texcoord);
    auto pdf_specular = kr * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    pdf_diffuse = 1.0 - pdf_specular;

    auto wo_local = ToLocal(wo, normal);
    auto pdf = pdf_diffuse * PdfHemisCos(wo_local);

    auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
    auto h = glm::normalize(-wi + wo);
    auto D = distrib->Pdf(h, normal);

    if (D > kEpsilon)
    {
        auto jacobian = std::abs(1.0 / (4.0 * glm::dot(wo, h)));
        pdf += pdf_specular * D * jacobian;
    }

    return pdf;
}

///\brief 是否映射纹理
bool RoughPlastic::TextureMapping() const
{
    return Material::TextureMapping() ||
           !diffuse_reflectance_->Constant() ||
           specular_reflectance_ && !specular_reflectance_->Constant();
}

///\brief 给定点是否透明
bool RoughPlastic::Transparent(const Vector2 &texcoord) const
{
    return Material::Transparent(texcoord) ||
           diffuse_reflectance_->Transparent(texcoord);
}

///\brief 补偿多次散射后又射出的光能
Float RoughPlastic::EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
{
    auto albedo_i = GetAlbedo(std::abs(cos_i_n));
    auto albedo_o = GetAlbedo(std::abs(cos_o_n));
    auto f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
    return f_ms * f_add_;
}

///\brief 获取给定点抽样镜面反射的权重
Float RoughPlastic::SpecularSamplingWeight(const Vector2 &texcoord) const
{
    if (specular_sampling_weight_ >= 0)
        return specular_sampling_weight_;

    auto kd = diffuse_reflectance_->Color(texcoord);
    auto d_sum = kd.r + kd.g + kd.b;
    if (!specular_reflectance_)
        return 3.0 / (d_sum + 3.0);

    auto ks = specular_reflectance_->Color(texcoord);
    auto s_sum = ks.r + ks.g + ks.b;
    return s_sum / (d_sum + s_sum);
}
NAMESPACE_END(simple_renderer)