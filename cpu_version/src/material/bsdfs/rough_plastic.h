#pragma once

#include "microfacet.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

class RoughPlastic : public Microfacet
{
public:
    /**
     * \brief 粗糙的塑料材质
     * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
     * \param alpha 粗糙度
     * \param int_ior 内折射率
     * \param ext_ior 外折射率
     * \param diffuse_reflectance 漫反射分量
     * \param nonlinear 是否考虑因内部散射而引起的非线性色移
     * \param specular_reflectance 镜面反射系数。注意，对于物理真实感绘制，应默认为 1。
     */
    RoughPlastic(MicrofacetDistribType distrib_type,
                 std::unique_ptr<Texture> alpha,
                 Float int_ior,
                 Float ext_ior,
                 std::unique_ptr<Texture> diffuse_reflectance,
                 bool nonlinear,
                 std::unique_ptr<Texture> specular_reflectance = nullptr)
        : Microfacet(MaterialType::kRoughPlastic,
                     distrib_type,
                     std::move(alpha),
                     nullptr),
          eta_(int_ior / ext_ior),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)),
          diffuse_reflectance_(std::move(diffuse_reflectance)),
          nonlinear_(nonlinear)
    {
        fdr_int_ = FresnelDiffuseReflectance(eta_inv_);
        fdr_ext_ = FresnelDiffuseReflectance(eta_);

        specular_sampling_weight_ = 0;
        if (diffuse_reflectance_->Constant() && (!specular_reflectance_ || specular_reflectance_->Constant()))
        {
            auto d_weight = diffuse_reflectance_->GetPixel(Vector2(0));
            auto d_sum = d_weight.r + d_weight.g + d_weight.b;
            Float s_sum = 3;
            if (specular_reflectance_)
            {
                auto s_weight = specular_reflectance_->GetPixel(Vector2(0));
                s_sum = s_weight.r + s_weight.g + s_weight.b;
            }
            specular_sampling_weight_ = s_sum / (d_sum + s_sum);
        }

        f_add_ = 0;
        if (!Microfacet::TextureMapping())
        {
            auto F_avg = AverageFresnelDielectric(eta_);
            f_add_ = Sqr(F_avg) * albedo_avg_ / (1 - F_avg * (1 - albedo_avg_));
        }
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        auto eta_inv = bs.inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);
        auto specular_sampling_weight = get_specular_sampling_weight(bs.texcoord);

        auto kr = Fresnel(-bs.wo, bs.normal, eta_inv);
        auto pdf_specular = kr * specular_sampling_weight,
             pdf_diffuse = (1 - kr) * (1 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

        auto sample_x = UniformFloat();
        if (sample_x < pdf_specular)
        {
            auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_u);
            auto [normal_micro, pdf] = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});
            bs.wi = -Reflect(-bs.wo, normal_micro);
            if (glm::dot(bs.wi, bs.normal) >= 0)
                return;
        }
        else
        {
            auto [wi_local, pdf] = HemisCos();
            bs.wi = -ToWorld(wi_local, bs.normal);
        }

        bs.pdf = Pdf(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
		if (bs.pdf < kEpsilonL)
		{
			bs.pdf = 0;
			return;
		}

		if (bs.get_weight)
			bs.weight = Eval(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
	
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {

        if (NotSameHemis(wo, normal))
            return Spectrum(0);

        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        auto [alpha_u, alpha_v] = GetAlpha(texcoord);

        auto cos_i_n = glm::dot(wi, normal);
        auto cos_o_n = glm::dot(wo, normal);

        auto fdr_int = !inside ? fdr_int_ : fdr_ext_,
             fdr_ext = !inside ? fdr_ext_ : fdr_int_;

        Spectrum albedo(0);

        auto diffuse_reflectance = get_diffuse_reflectance(texcoord);
        if (nonlinear_)
        {
            albedo = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_int);
        }
        else
        {
            albedo = diffuse_reflectance / (1 - fdr_int);
        }

        auto kr_i = Fresnel(wi, normal, eta_inv);
        auto kr_o = Fresnel(-wo, normal, eta_inv);
        albedo *= Sqr(eta_inv) * (1 - kr_i) * (1 - kr_o) * kPiInv;

        auto h = glm::normalize(-wi + wo);
        auto F = Fresnel(wi, h, eta_inv);

        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_u);
        auto D = distrib->Pdf(h, normal);
        if (D > kEpsilon)
        {
            auto G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);
            auto value = F * D * G / (4 * std::fabs(cos_i_n * cos_o_n));
            if (!Microfacet::TextureMapping() && albedo_avg_ < kOneMinusEpsilon)
                value += EvalMultipleScatter(cos_i_n, cos_o_n);

            auto specular_reflectance = Spectrum(1);
            if (specular_reflectance_)
            {
                if (texcoord != nullptr)
                    specular_reflectance = specular_reflectance_->GetPixel(*texcoord);
                else
                    specular_reflectance = specular_reflectance_->GetPixel(Vector2(0));
            }
            albedo += specular_reflectance * value;
        }

        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return 0;

        auto [alpha_u, alpha_v] = GetAlpha(texcoord);
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        auto kr = Fresnel(wi, normal, eta_inv);
        auto specular_sampling_weight = get_specular_sampling_weight(texcoord);

        auto pdf_specular = kr * specular_sampling_weight,
             pdf_diffuse = (1 - kr) * (1 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        pdf_diffuse = 1 - pdf_specular;

        auto wo_local = ToLocal(wo, normal);
        auto result = pdf_diffuse * PdfHemisCos(wo_local);

        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
        auto h = glm::normalize(-wi + wo);
        auto D = distrib->Pdf(h, normal);

        if (D > kEpsilon)
        {
            auto jacobian = std::fabs(1 / (4 * glm::dot(wo, h)));
            result += pdf_specular * D * jacobian;
        }
        return result;
    }

	///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Microfacet::TextureMapping() ||
               !diffuse_reflectance_->Constant() ||
               (specular_reflectance_ && !specular_reflectance_->Constant());
    }

	///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Material::Transparent(texcoord) ||
               diffuse_reflectance_->Transparent(texcoord);
    }

private:
    bool nonlinear_; // 是否考虑因内部散射而引起的非线性色移
    Float eta_;      // 光线射入材质的相对折射率
    Float eta_inv_;  // 光线射出材质的相对折射率
    Float fdr_ext_;
    Float fdr_int_;
    Float specular_sampling_weight_;                // 抽样镜面反射权重
    Float f_add_;                                   //补偿多次散射后出射光能的系数
    std::unique_ptr<Texture> specular_reflectance_; // 镜面反射系数。注意，对于物理真实感绘制默认为 1，应为空指针。
    std::unique_ptr<Texture> diffuse_reflectance_;  // 漫反射系数

	///\brief 补偿多次散射后又射出的光能
    Float EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
    {
        auto albedo_i = GetAlbedo(std::fabs(cos_i_n));
        auto albedo_o = GetAlbedo(std::fabs(cos_o_n));
        auto f_ms = (1 - albedo_o) * (1 - albedo_i) / (kPi * (1 - albedo_avg_));
        return f_ms * f_add_;
    }

    ///\brief 获取给定点的漫反射系数
    Spectrum get_diffuse_reflectance(const Vector2 *texcoord) const
    {
        if (diffuse_reflectance_->Constant())
            return diffuse_reflectance_->GetPixel(Vector2(0));
        else
            return diffuse_reflectance_->GetPixel(*texcoord);
    }

    ///\brief 获取给定点抽样镜面反射的权重
    Float get_specular_sampling_weight(const Vector2 *texcoord) const
    {
        if (diffuse_reflectance_->Constant() && specular_reflectance_)
            return specular_sampling_weight_;

        auto kd = get_diffuse_reflectance(texcoord);
        auto d_sum = kd.r + kd.g + kd.b;
        if (!specular_reflectance_)
            return 3 / (d_sum + 3);

        auto s_weight = specular_reflectance_->GetPixel(Vector2(0));
        auto s_sum = s_weight.r + s_weight.g + s_weight.b;
        return s_sum / (d_sum + s_sum);
    }
};

NAMESPACE_END(simple_renderer)