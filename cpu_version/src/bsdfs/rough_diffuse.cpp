#include "rough_diffuse.h"

NAMESPACE_BEGIN(simple_renderer)

/* Conversion from Beckmann-style RMS roughness to
   Oren-Nayar-style slope-area variance. The factor
   of 1/sqrt(2) was found to be a perfect fit up
   to extreme roughness values (>.5), after which
   the match is not as good anymore */
const Float kConversionFactor = 1.0 / std::sqrt(2.0);

///\brief 粗糙的理想漫反射材质
RoughDiffuse::RoughDiffuse(std::unique_ptr<Texture> reflectance,
                           std::unique_ptr<Texture> alpha,
                           bool use_fast_approx)
    : Material(MaterialType::kRoughDiffuse),
      reflectance_(std::move(reflectance)),
      alpha_(std::move(alpha)),
      use_fast_approx_(use_fast_approx) {}

///\brief 根据光线出射方向和表面法线方向抽样光线入射方向，法线方向已被处理至与光线出射方向夹角大于90度
void RoughDiffuse::Sample(BsdfSampling &bs) const
{
    auto [wi_local, pdf] = HemisCos();
    if (pdf < kEpsilonL)
        return;

    bs.wi = -ToWorld(wi_local, bs.normal);
    bs.pdf = pdf;

    if (!bs.get_attenuation)
        return;

    bs.attenuation = Eval(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
}

///\brief 根据光线入射方向、出射方向和表面法线方向，计算 BSDF 权重，法线方向已被处理至与光线入射方向夹角大于90度
Spectrum RoughDiffuse::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    auto sigma = alpha_->Color(texcoord).x * kConversionFactor;
    auto sigma_2 = sigma * sigma;

    auto cos_theta_i = glm::dot(-wi, normal),
         cos_theta_o = glm::dot(wo, normal);
    auto sin_theta_i = std::sqrt(1.0 - cos_theta_i * cos_theta_i),
         sin_theta_o = std::sqrt(1.0 - cos_theta_o * cos_theta_o);

    auto wi_local = ToLocal(-wi, normal);
    auto phi_i = static_cast<Float>(0),
         theta_i = static_cast<Float>(0);
    CartesianToSpherical(wi_local, theta_i, phi_i);

    auto wo_local = ToLocal(wo, normal);
    auto phi_o = static_cast<Float>(0),
         theta_o = static_cast<Float>(0);
    CartesianToSpherical(wo_local, theta_o, phi_o);

    auto cos_phi_diff = static_cast<Float>(std::cos(phi_i) * std::cos(phi_o) +
                                           std::sin(phi_i) * std::sin(phi_o));

    if (use_fast_approx_)
    {
        auto A = 1.0 - 0.5 * sigma_2 / (sigma_2 + 0.33),
             B = 0.45 * sigma_2 / (sigma_2 + 0.09);
        auto sin_alpha = static_cast<Float>(0),
             tan_beta = static_cast<Float>(0);
        if (cos_theta_i > cos_theta_o)
        {
            sin_alpha = sin_theta_o;
            tan_beta = sin_theta_i / cos_theta_i;
        }
        else
        {
            sin_alpha = sin_theta_i;
            tan_beta = sin_theta_o / cos_theta_o;
        }
        return reflectance_->Color(texcoord) *
               kPiInv *
               (A + B * std::max(cos_phi_diff, (Float)0) * sin_alpha * tan_beta);
    }
    else
    {
        auto alpha = std::max(theta_i, theta_o),
             beta = std::min(theta_i, theta_o);
        auto sin_alpha = static_cast<Float>(0),
             sin_beta = static_cast<Float>(0),
             tan_beta = static_cast<Float>(0);
        if (cos_theta_i > cos_theta_o)
        {
            sin_alpha = sin_theta_o;
            sin_beta = sin_theta_i;
            tan_beta = sin_theta_i / cos_theta_i;
        }
        else
        {
            sin_alpha = sin_theta_i;
            sin_beta = sin_theta_o;
            tan_beta = sin_theta_o / cos_theta_o;
        }

        auto tmp = static_cast<Float>(sigma_2 / (sigma_2 + 0.09)),
             tmp2 = static_cast<Float>((4.0 * kPiInv * kPiInv) * alpha * beta),
             tmp3 = static_cast<Float>(2.0 * beta * kPiInv);

        auto C1 = static_cast<Float>(1.0 - 0.5 * sigma_2 / (sigma_2 + 0.33)),
             C2 = static_cast<Float>(0.45 * tmp),
             C3 = static_cast<Float>(0.125 * tmp * tmp2 * tmp2),
             C4 = static_cast<Float>(0.17 * sigma_2 / (sigma_2 + 0.13));

        if (cos_phi_diff > 0)
            C2 *= sin_alpha;
        else
            C2 *= sin_alpha - tmp3 * tmp3 * tmp3;

        auto tan_half = (sin_alpha + sin_beta) / (std::sqrt(std::max(0.0, 1.0 - sin_alpha * sin_alpha)) +
                                                  std::sqrt(std::max(0.0, 1.0 - sin_beta * sin_beta)));

        auto rho = reflectance_->Color(texcoord),
             sngl_scat = rho * (C1 + cos_phi_diff * C2 * tan_beta +
                                (1.0 - std::abs(cos_phi_diff)) * C3 * tan_half),
             dbl_scat = rho * rho * (C4 * (1.0 - cos_phi_diff * tmp3 * tmp3));
        return (sngl_scat + dbl_scat) * kPiInv;
    }
}

///\brief 根据光线入射方向和表面法线方向，计算光线从给定出射方向射出的概率，法线方向已被处理至与光线入射方向夹角大于90度
Float RoughDiffuse::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    // 表面法线方向，光线入射和出射需在介质同侧
    if (NotSameHemis(wo, normal))
        return 0;

    auto wo_local = ToLocal(wo, normal);
    return PdfHemisCos(wo_local);
}

///\brief 是否映射纹理
bool RoughDiffuse::TextureMapping() const
{
    return Material::TextureMapping() ||
           !reflectance_->Constant() ||
           !alpha_->Constant();
}

///\brief 给定点是否透明
bool RoughDiffuse::Transparent(const Vector2 &texcoord) const
{
    return Material::Transparent(texcoord) ||
           reflectance_->Transparent(texcoord);
}

NAMESPACE_END(simple_renderer)