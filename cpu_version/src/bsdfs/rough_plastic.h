#pragma once

#include "microfacet.h"
#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 粗糙的塑料材质派生类
class RoughPlastic : public Material, public Microfacet
{
public:
    ///\brief 粗糙的塑料材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param diffuse_reflectance 漫反射分量
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ///\param distrib_type 用于模拟表面粗糙度的微表面分布的类型
    ///\param alpha 材质的粗糙度
    ///\param nonlinear 是否考虑因内部散射而引起的非线性色移
    RoughPlastic(Float int_ior, Float ext_ior,
                 std::unique_ptr<Texture> diffuse_reflectance,
                 std::unique_ptr<Texture> specular_reflectance,
                 MicrofacetDistribType distrib_type,
                 std::unique_ptr<Texture> alpha, bool nonlinear)
        : Material(MaterialType::kRoughPlastic),
          Microfacet(distrib_type, std::move(alpha), nullptr),
          eta_inv_(ext_ior / int_ior), fdr_(AverageFresnel(int_ior / ext_ior)),
          diffuse_reflectance_(std::move(diffuse_reflectance)),
          specular_reflectance_(std::move(specular_reflectance)),
          nonlinear_(nonlinear), specular_sampling_weight_(-1), f_add_(0)
    {
        if (diffuse_reflectance_->Constant() &&
            (!specular_reflectance_ || specular_reflectance_->Constant()))
        {
            Spectrum kd = diffuse_reflectance_->Color(Vector2(0));
            Float s_sum = 3.0, d_sum = kd.r + kd.g + kd.b;
            if (specular_reflectance_)
            {
                Spectrum ks = specular_reflectance_->Color(Vector2(0));
                s_sum = ks.r + ks.g + ks.b;
            }
            specular_sampling_weight_ = s_sum / (d_sum + s_sum);
        }
        if (Material::TextureMapping())
            return;
        ComputeAlbedoTable();
        if (albedo_avg_ < 0)
            return;
        f_add_ = Sqr(fdr_) * albedo_avg_ / (1.0 - fdr_ * (1.0 - albedo_avg_));
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        Float D = 0,
              kr_o = Fresnel(-bs.wo, bs.normal, eta_inv_),
              specular_sampling_weight = SpecularSamplingWeight(bs.texcoord),
              pdf_specular = kr_o * specular_sampling_weight,
              pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);
        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_u);
        auto h = Vector3(0);
        if (UniformFloat() < pdf_specular)
        {
            std::tie(h, D) = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});
            bs.wi = -Reflect(-bs.wo, h);
            if (glm::dot(bs.wi, bs.normal) >= 0)
                return;
        }
        else
        {
            SampleHemisCos(bs.normal, bs.wi);
            h = glm::normalize(-bs.wi + bs.wo);
            D = distrib->Pdf(h, bs.normal);
        }
        Float kr_i = Fresnel(bs.wi, bs.normal, eta_inv_);
        pdf_specular = kr_i * specular_sampling_weight,
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        bs.pdf = (1.0 - pdf_specular) * PdfHemisCos(bs.wo, bs.normal);
        if (D > kEpsilonPdf)
            bs.pdf += pdf_specular * D * std::abs(1.0 / (4.0 * glm::dot(bs.wo, h)));
        if (bs.pdf < kEpsilonPdf || !bs.get_attenuation)
            return;
        Spectrum diffuse_reflectance = diffuse_reflectance_->Color(bs.texcoord);
        if (nonlinear_)
            bs.attenuation = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
        else
            bs.attenuation = diffuse_reflectance / (1.0 - fdr_);
        bs.attenuation *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        if (D > kEpsilonPdf)
        {
            Float F = Fresnel(bs.wi, h, eta_inv_),
                  G = distrib->SmithG1(-bs.wi, h, bs.normal) * distrib->SmithG1(bs.wo, h, bs.normal),
                  cos_theta_i = glm::dot(bs.wi, bs.normal),
                  cos_theta_o = glm::dot(bs.wo, bs.normal),
                  attenuation = F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o));
            if (albedo_avg_ > 0)
                attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
            Spectrum specular_reflectance = specular_reflectance_ ? specular_reflectance_->Color(bs.texcoord) : Spectrum(1);
            bs.attenuation += specular_reflectance * attenuation;
        }
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
                  bool inside) const override
    {
        auto albedo = Spectrum(0);
        Spectrum diffuse_reflectance = diffuse_reflectance_->Color(texcoord);
        if (nonlinear_)
            albedo = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
        else
            albedo = diffuse_reflectance / (1.0 - fdr_);
        Float kr_i = Fresnel(wi, normal, eta_inv_),
              kr_o = Fresnel(-wo, normal, eta_inv_);
        albedo *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        auto [alpha_u, alpha_v] = GetAlpha(texcoord);
        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_u);
        Vector3 h = glm::normalize(-wi + wo);
        Float D = distrib->Pdf(h, normal),
              F = Fresnel(wi, h, eta_inv_),
              cos_theta_i = glm::dot(wi, normal),
              cos_theta_o = glm::dot(wo, normal);
        if (D > kEpsilonPdf)
        {
            Float G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal),
                  attenuation = F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o));
            if (albedo_avg_ > 0)
                attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
            Spectrum specular_reflectance = specular_reflectance_ ? specular_reflectance_->Color(texcoord) : Spectrum(1);
            albedo += specular_reflectance * attenuation;
        }
        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
              bool inside) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(wo, normal))
            return 0;
        Float kr = Fresnel(wi, normal, eta_inv_),
              specular_sampling_weight = SpecularSamplingWeight(texcoord),
              pdf_specular = kr * specular_sampling_weight,
              pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        Float pdf = (1.0 - pdf_specular) * PdfHemisCos(wo, normal);
        auto [alpha_u, alpha_v] = GetAlpha(texcoord);
        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
        Vector3 h = glm::normalize(-wi + wo);
        Float D = distrib->Pdf(h, normal);
        if (D > kEpsilonPdf)
            pdf += pdf_specular * D * std::abs(1.0 / (4.0 * glm::dot(wo, h)));
        return pdf;
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               !diffuse_reflectance_->Constant() ||
               specular_reflectance_ && !specular_reflectance_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Material::Transparent(texcoord) ||
               diffuse_reflectance_->Transparent(texcoord);
    }

private:
    ///\brief 补偿多次散射后又射出的光能
    Float EvalMultipleScatter(Float cos_theta_i, Float cos_theta_o) const
    {
        Float albedo_i = GetAlbedo(std::abs(cos_theta_i)),
              albedo_o = GetAlbedo(std::abs(cos_theta_o)),
              f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
        return f_ms * f_add_;
    }

    ///\brief 获取给定点抽样镜面反射的权重
    Float SpecularSamplingWeight(const Vector2 &texcoord) const
    {
        if (specular_sampling_weight_ >= 0)
            return specular_sampling_weight_;

        Spectrum kd = diffuse_reflectance_->Color(texcoord);
        Float d_sum = kd.r + kd.g + kd.b;
        if (!specular_reflectance_)
            return 3.0 / (d_sum + 3.0);

        Spectrum ks = specular_reflectance_->Color(texcoord);
        Float s_sum = ks.r + ks.g + ks.b;
        return s_sum / (d_sum + s_sum);
    }

    Float eta_inv_;                                 //外部折射率与介质折射率之比
    std::unique_ptr<Texture> diffuse_reflectance_;  //漫反射系数
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    bool nonlinear_;                                //是否考虑因内部散射而引起的非线性色移
    Float specular_sampling_weight_;                //抽样镜面反射权重
    Float fdr_;                                     //漫反射菲涅尔项平均值
    Float f_add_;                                   //补偿多次散射后出射光能的系数
};

NAMESPACE_END(raytracer)