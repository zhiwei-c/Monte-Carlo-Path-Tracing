#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 粗糙的理想漫反射材质派生类
class RoughDiffuse : public Material
{
public:
    ///\brief 粗糙的理想漫反射材质，源于 Oren–Nayar Reflectance Model
    ///\param id 材质id
    ///\param reflectance 漫反射系数
    RoughDiffuse(std::unique_ptr<Texture> reflectance,
                 std::unique_ptr<Texture> alpha, bool use_fast_approx)
        : Material(MaterialType::kRoughDiffuse),
          reflectance_(std::move(reflectance)),
          alpha_(std::move(alpha)), use_fast_approx_(use_fast_approx)
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        SampleHemisCos(bs.normal, bs.wi, &bs.pdf);
        if (!bs.get_attenuation)
            return;
        bs.attenuation = Eval(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
                  bool inside) const override
    {
        /* Conversion from Beckmann-style RMS roughness to
           Oren-Nayar-style slope-area variance. The factor
           of 1/sqrt(2) was found to be a perfect fit up
           to extreme roughness values (>.5), after which
           the match is not as good anymore */
        Float conversion_factor = 1.0 / std::sqrt(2.0);
        Float sigma = alpha_->Color(texcoord).x * conversion_factor,
              sigma_2 = sigma * sigma;
        Float cos_theta_i = glm::dot(-wi, normal),
              cos_theta_o = glm::dot(wo, normal),
              sin_theta_i = std::sqrt(1.0 - cos_theta_i * cos_theta_i),
              sin_theta_o = std::sqrt(1.0 - cos_theta_o * cos_theta_o);
        Float phi_i = 0, theta_i = 0;
        CartesianToSpherical(ToLocal(-wi, normal), theta_i, phi_i);
        Float phi_o = 0, theta_o = 0;
        CartesianToSpherical(ToLocal(wo, normal), theta_o, phi_o);
        Float cos_phi_diff = std::cos(phi_i) * std::cos(phi_o) + std::sin(phi_i) * std::sin(phi_o);
        if (use_fast_approx_)
        {
            Float A = 1.0 - 0.5 * sigma_2 / (sigma_2 + 0.33),
                  B = 0.45 * sigma_2 / (sigma_2 + 0.09);
            Float sin_alpha = 0, tan_beta = 0;
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
            return reflectance_->Color(texcoord) * kPiInv *
                   (A + B * std::max(cos_phi_diff, 0.0) * sin_alpha * tan_beta);
        }
        else
        {
            Float alpha = std::max(theta_i, theta_o),
                  beta = std::min(theta_i, theta_o);
            Float sin_alpha = 0, sin_beta = 0, tan_beta = 0;
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
            Float tmp = sigma_2 / (sigma_2 + 0.09),
                  tmp2 = (4.0 * kPiInv * kPiInv) * alpha * beta,
                  tmp3 = 2.0 * beta * kPiInv;
            Float C1 = 1.0 - 0.5 * sigma_2 / (sigma_2 + 0.33),
                  C2 = 0.45 * tmp,
                  C3 = 0.125 * tmp * tmp2 * tmp2,
                  C4 = 0.17 * sigma_2 / (sigma_2 + 0.13);
            if (cos_phi_diff > 0)
                C2 *= sin_alpha;
            else
                C2 *= sin_alpha - tmp3 * tmp3 * tmp3;
            Float tan_half = (sin_alpha + sin_beta) / (std::sqrt(std::max(0.0, 1.0 - sin_alpha * sin_alpha)) +
                                                       std::sqrt(std::max(0.0, 1.0 - sin_beta * sin_beta)));
            Spectrum rho = reflectance_->Color(texcoord),
                     sngl_scat = rho * (C1 + cos_phi_diff * C2 * tan_beta +
                                        (1.0 - std::abs(cos_phi_diff)) * C3 * tan_half),
                     dbl_scat = rho * rho * (C4 * (1.0 - cos_phi_diff * tmp3 * tmp3));
            return (sngl_scat + dbl_scat) * kPiInv;
        }
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
              bool inside) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(wo, normal))
            return 0;
        return PdfHemisCos(wo, normal);
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               !reflectance_->Constant() ||
               !alpha_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Material::Transparent(texcoord) ||
               reflectance_->Transparent(texcoord);
    }

private:
    std::unique_ptr<Texture> reflectance_; //漫反射系数
    std::unique_ptr<Texture> alpha_;       //表面粗糙程度
    bool use_fast_approx_;                 //是否快速近似
};

NAMESPACE_END(raytracer)