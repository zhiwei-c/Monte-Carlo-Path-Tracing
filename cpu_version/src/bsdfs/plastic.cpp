#include "plastic.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 平滑的塑料材质
Plastic::Plastic(Float int_ior,
                 Float ext_ior,
                 std::unique_ptr<Texture> diffuse_reflectance,
                 std::unique_ptr<Texture> specular_reflectance,
                 bool nonlinear)
    : Material(MaterialType::kPlastic),
      eta_inv_(ext_ior / int_ior),
      fdr_(AverageFresnel(int_ior / ext_ior)),
      diffuse_reflectance_(std::move(diffuse_reflectance)),
      specular_reflectance_(std::move(specular_reflectance)),
      nonlinear_(nonlinear)
{
    specular_sampling_weight_ = -1;

    if (!diffuse_reflectance_->Constant() ||
        specular_reflectance_ && !specular_reflectance_->Constant())
        return;

    auto kd = diffuse_reflectance_->Color(Vector2(0));
    auto d_sum = kd.r + kd.g + kd.b;
    auto s_sum = static_cast<Float>(3);
    if (specular_reflectance_)
    {
        auto ks = specular_reflectance_->Color(Vector2(0));
        s_sum = ks.r + ks.g + ks.b;
    }

    specular_sampling_weight_ = s_sum / (d_sum + s_sum);
}

///\brief 根据光线出射方向和表面法线方向抽样光线入射方向，法线方向已被处理至与光线出射方向夹角大于90度
void Plastic::Sample(BsdfSampling &bs) const
{
    auto kr_o = Fresnel(-bs.wo, bs.normal, eta_inv_);
    auto specular_sampling_weight = SpecularSamplingWeight(bs.texcoord);
    auto pdf_specular = kr_o * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

    bs.pdf = 0;
    auto kr_i = static_cast<Float>(0);
    auto specular = false;
    if (UniformFloat() < pdf_specular)
    {
        bs.wi = -Reflect(-bs.wo, bs.normal);
        kr_i = kr_o;
        bs.pdf += pdf_specular;
        specular = true;
    }
    else
    {
        auto [wi_local, pdf] = HemisCos();
        bs.wi = -ToWorld(wi_local, bs.normal);
        kr_i = Fresnel(bs.wi, bs.normal, eta_inv_);
        pdf_specular = kr_i * specular_sampling_weight;
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    }

    pdf_diffuse = 1.0 - pdf_specular;
    bs.pdf += pdf_diffuse * PdfHemisCos(ToLocal(bs.wo, bs.normal));
    if (bs.pdf < kEpsilonL)
    {
        bs.pdf = 0;
        return;
    };

    if (!bs.get_attenuation)
        return;

    auto diffuse_reflectance = diffuse_reflectance_->Color(bs.texcoord);
    if (nonlinear_)
        bs.attenuation = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
    else
        bs.attenuation = diffuse_reflectance / (1.0 - fdr_);
    bs.attenuation *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;

    if (specular)
        bs.attenuation += kr_i * (specular_reflectance_ ? specular_reflectance_->Color(bs.texcoord) : Spectrum(1));
}

///\brief 根据光线入射方向、出射方向和表面法线方向，计算 BSDF 权重，法线方向已被处理至与光线入射方向夹角大于90度
Spectrum Plastic::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    auto albedo = Spectrum(0);
    auto diffuse_reflectance = diffuse_reflectance_->Color(texcoord);
    if (nonlinear_)
        albedo = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
    else
        albedo = diffuse_reflectance / (1.0 - fdr_);

    auto kr_i = Fresnel(wi, normal, eta_inv_);
    auto kr_o = Fresnel(-wo, normal, eta_inv_);
    albedo *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;

    if (SameDirection(Reflect(wi, normal), wo))
        albedo += kr_i * (specular_reflectance_ ? specular_reflectance_->Color(texcoord) : Spectrum(1));

    return albedo;
}

///\brief 根据光线入射方向和表面法线方向，计算光线从给定出射方向射出的概率，法线方向已被处理至与光线入射方向夹角大于90度
Float Plastic::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    // 表面法线方向，光线入射和出射需在介质同侧
    if (NotSameHemis(wo, normal))
        return 0;

    auto kr = Fresnel(wi, normal, eta_inv_);
    auto specular_sampling_weight = SpecularSamplingWeight(texcoord);
    auto pdf_specular = kr * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    pdf_diffuse = 1.0 - pdf_specular;
    auto wo_local = ToLocal(wo, normal);
    auto pdf_diffuse_local = PdfHemisCos(wo_local);
    auto pdf = pdf_diffuse * pdf_diffuse_local;

    if (SameDirection(wo, Reflect(wi, normal)))
        pdf += pdf_specular;

    return pdf;
}

///\brief 是否映射纹理
bool Plastic::TextureMapping() const
{
    return Material::TextureMapping() ||
           !diffuse_reflectance_->Constant() ||
           specular_reflectance_ && !specular_reflectance_->Constant();
}

///\brief 给定点是否透明
bool Plastic::Transparent(const Vector2 &texcoord) const
{
    return Material::Transparent(texcoord) ||
           diffuse_reflectance_->Transparent(texcoord);
}

///\brief 获取给定点抽样镜面反射的权重
Float Plastic::SpecularSamplingWeight(const Vector2 &texcoord) const
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