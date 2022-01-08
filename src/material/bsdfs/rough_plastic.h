#pragma once

#include "microfacet.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

class RoughPlastic : public Microfacet
{
public:
    /**
	 * \brief 粗糙的塑料材质
	 * \param id 材质id
	 * \param diffuse_reflectance 漫反射分量
	 * \param diffuse_map  漫反射纹理
	 * \param nonlinear 是否考虑因内部散射而引起的非线性色移
	 * \param ext_ior 外折射率
	 * \param int_ior 内折射率
	 * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	 * \param alpha 粗糙度
	 * \param specular_reflectance 镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。
	*/
    RoughPlastic(const std::string &id,
                 const Vector3 &diffuse_reflectance,
                 Texture *diffuse_map,
                 bool nonlinear,
                 Float ext_ior,
                 Float int_ior,
                 MicrofacetDistribType distrib_type,
                 Float alpha,
                 const Vector3 &specular_reflectance = Vector3(1.f))
        : Microfacet(id,
                     MaterialType::kRoughPlastic,
                     distrib_type,
                     alpha,
                     alpha),
          diffuse_reflectance_(diffuse_reflectance),
          diffuse_map_(diffuse_map),
          nonlinear_(nonlinear),
          eta_(int_ior / ext_ior),
          eta_inv_(ext_ior / int_ior),
          alpha_(alpha),
          specular_reflectance_(specular_reflectance)
    {
        fdr_int_ = FresnelDiffuseReflectance(eta_inv_);
        fdr_ext_ = FresnelDiffuseReflectance(eta_);
        auto d_sum = diffuse_reflectance_.r + diffuse_reflectance_.g + diffuse_reflectance_.b;
        s_sum_ = specular_reflectance_.r + specular_reflectance_.g + specular_reflectance_.b;
        specular_sampling_weight_ = s_sum_ / (d_sum + s_sum_);

        auto F_avg = AverageFresnelDielectric(eta_);
        f_add_ = Sqr(F_avg) * albedo_avg_ / (1 - F_avg * (1 - albedo_avg_));
    }

    ~RoughPlastic()
    {
        if (diffuse_map_)
            DeleteTexturePointer(diffuse_map_);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    std::pair<Vector3, BsdfSamplingType> Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto specular_sampling_weight = specular_sampling_weight_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
            {
                auto kd = diffuse_map_->GetPixel(*texcoord);
                auto d_sum = kd.r + kd.g + kd.b;
                specular_sampling_weight = s_sum_ / (d_sum + s_sum_);
            }
        }

        const auto &wi_pseudo = -wo;
        auto kr_pseudo = Fresnel(wi_pseudo, normal, eta_inv);
        auto pdf_specular = kr_pseudo * specular_sampling_weight,
             pdf_diffuse = (1 - kr_pseudo) * (1 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        pdf_diffuse = 1 - pdf_specular;

        auto sample_x = UniformFloat();
        if (sample_x < pdf_specular)
        {
            auto distrib = InitDistrib(distrib_type_, alpha_, alpha_);
            auto normal_micro = distrib->Sample(normal, {UniformFloat(), UniformFloat()});
            DeleteDistribPointer(distrib);

            return {-Reflect(wi_pseudo, normal_micro), BsdfSamplingType::kSpecularReflection};
        }
        else
        {
            auto wo_pseudo_local = HemisCos();
            return {-ToWorld(wo_pseudo_local, normal), BsdfSamplingType::kReflection};
        }
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
    {

        if (NotSameHemis(wo, normal))
            return Vector3(0);

        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto cos_i_n = glm::dot(wi, normal);
        auto cos_o_n = glm::dot(wo, normal);

        auto fdr_int = !inside ? fdr_int_ : fdr_ext_,
             fdr_ext = !inside ? fdr_ext_ : fdr_int_;

        Vector3 weight(0);

        auto diffuse_reflectance = diffuse_reflectance_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
                diffuse_reflectance = diffuse_map_->GetPixel(*texcoord);
        }
        if (nonlinear_)
        {
            weight = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_int);
        }
        else
        {
            weight = diffuse_reflectance / (1 - fdr_int);
        }

        auto kr_i = Fresnel(wi, normal, eta_inv);
        auto kr_o = Fresnel(-wo, normal, eta_inv);
        weight *= Sqr(eta_inv) * (1 - kr_i) * (1 - kr_o) * kPiInv;

        auto h = glm::normalize(-wi + wo);
        auto F = Fresnel(wi, h, eta_inv);

        auto distrib = InitDistrib(distrib_type_, alpha_, alpha_);
        auto D = distrib->Eval(h, normal);
        auto G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);
        DeleteDistribPointer(distrib);

        auto value = F * D * G / (4 * std::fabs(cos_i_n * cos_o_n));
        auto weight_loss = EvalMultipleScatter(cos_i_n, cos_o_n);
        value += weight_loss;

        weight += specular_reflectance_ * value;

        return weight;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
    {
        if (NotSameHemis(wo, normal))
            return 0;

        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        auto kr = Fresnel(wi, normal, eta_inv);
        auto specular_sampling_weight = specular_sampling_weight_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
            {
                auto kd = diffuse_map_->GetPixel(*texcoord);
                auto d_sum = kd.r + kd.g + kd.b;
                specular_sampling_weight = s_sum_ / (d_sum + s_sum_);
            }
        }
        auto pdf_specular = kr * specular_sampling_weight,
             pdf_diffuse = (1 - kr) * (1 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        pdf_diffuse = 1 - pdf_specular;

        auto wo_local = ToLocal(wo, normal);
        auto result = pdf_diffuse * PdfHemisCos(wo_local);

        auto distrib = InitDistrib(distrib_type_, alpha_, alpha_);
        auto h = glm::normalize(-wi + wo);
        auto pdf_normal_micro = distrib->Eval(h, normal);
        DeleteDistribPointer(distrib);

        auto jacobian = std::fabs(1 / (4 * glm::dot(wo, h)));
        result += pdf_specular * pdf_normal_micro * jacobian;

        return result;
    }

    bool TextureMapping() const override { return diffuse_map_ != nullptr; }

    bool Transparent(const Vector2 &texcoord) const override
    {
        if (Material::Transparent(texcoord))
            return true;
        else if (diffuse_map_)
            return diffuse_map_->Transparent(texcoord);
        else
            return false;
    }

private:
    Vector3 diffuse_reflectance_;        // 漫反射系数
    Texture *diffuse_map_;               // 漫反射纹理
    bool nonlinear_;                     // 是否考虑因内部散射而引起的非线性色移
    Float eta_;                          // 光线射入材质的相对折射率
    Float eta_inv_;                      // 光线射出材质的相对折射率
    Vector3 specular_reflectance_;       // 镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。
    MicrofacetDistribType distrib_type_; // 用于模拟表面粗糙度的微表面分布的类型
    Float alpha_;                        // 粗糙度

    Float fdr_ext_;
    Float fdr_int_;
    Float specular_sampling_weight_;
    Float s_sum_;
    Float f_add_;

    Float EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
    {
        auto albedo_i = GetAlbedo(cos_i_n);
        auto albedo_o = GetAlbedo(cos_o_n);
        auto f_ms = (1 - albedo_o) * (1 - albedo_i) / (kPi * (1 - albedo_avg_));
        return f_ms * f_add_;
    }
};

NAMESPACE_END(simple_renderer)