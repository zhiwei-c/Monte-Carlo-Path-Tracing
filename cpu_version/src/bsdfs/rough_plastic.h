#pragma once

#include "microfacet.h"
#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 粗糙的塑料材质派生类
class RoughPlastic : public Bsdf, public Microfacet
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
    RoughPlastic(Float int_ior, Float ext_ior, std::unique_ptr<Texture> diffuse_reflectance,
                 std::unique_ptr<Texture> specular_reflectance, MicrofacetDistribType distrib_type,
                 std::unique_ptr<Texture> alpha, bool nonlinear)
        : Bsdf(BsdfType::kRoughPlastic), Microfacet(distrib_type, std::move(alpha), nullptr),
          fdr_(AverageFresnel(int_ior / ext_ior)), eta_inv_(ext_ior / int_ior), nonlinear_(nonlinear),
          diffuse_reflectance_(std::move(diffuse_reflectance)), specular_reflectance_(std::move(specular_reflectance)),
          specular_sampling_weight_(-1), f_add_(0)
    {
        if (diffuse_reflectance_->Constant() && (!specular_reflectance_ || specular_reflectance_->Constant()))
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
        if (Bsdf::TextureMapping())
            return;
        ComputeAlbedoTable();
        if (albedo_avg_ < 0)
            return;
        f_add_ = Sqr(fdr_) * albedo_avg_ / (1.0 - fdr_ * (1.0 - albedo_avg_));
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(SamplingRecord &rec) const override
    {
        Float kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_),                 //出射菲涅尔项
            specular_sampling_weight = SpecularSamplingWeight(rec.texcoord), //抽样镜面反射的权重
            pdf_specular = kr_o * specular_sampling_weight,                  //抽样镜面反射分量的概率
            pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);   //抽样漫反射分量的概率
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        auto [alpha_u, alpha_v] = GetAlpha(rec.texcoord);            //景物表面沿切线方向和副切线方向的粗糙程度
        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_u); //微表面分布
        Float D = 0;                                                 //微表面法线分布概率（相对于宏观表面法线）
        auto h = Vector3(0);                                         //微表面法线
        if (UniformFloat() < pdf_specular)
        { //从镜面反射分量抽样光线方向
            std::tie(h, D) = distrib->Sample(rec.normal, {UniformFloat(), UniformFloat()});
            rec.wi = -Reflect(-rec.wo, h);
        }
        else
        { //从漫反射分量抽样光线方向
            SampleHemisCos(rec.normal, rec.wi);
            h = glm::normalize(-rec.wi + rec.wo);
            D = distrib->Pdf(h, rec.normal);
        }
        Float cos_theta_i = glm::dot(-rec.wi, rec.normal); //入射光线方向和宏观表面法线方向夹角的余弦
        if (cos_theta_i < kEpsilon)
            return;
        Float kr_i = Fresnel(rec.wi, rec.normal, eta_inv_); //入射菲涅尔项
        pdf_specular = kr_i * specular_sampling_weight,
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec.pdf = (1.0 - pdf_specular) * PdfHemisCos(rec.wo, rec.normal);
        if (D > kEpsilonPdf)
            rec.pdf += pdf_specular * D * std::abs(1.0 / (4.0 * glm::dot(rec.wo, h)));
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kReflect;
        //计算光能衰减系数
        if (!rec.get_attenuation)
            return;
        Spectrum diffuse_reflectance = diffuse_reflectance_->Color(rec.texcoord);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_);
        rec.attenuation *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        if (D > kEpsilonPdf)
        {
            Float F = Fresnel(rec.wi, h, eta_inv_),                                                     //菲涅尔项
                G = distrib->SmithG1(-rec.wi, h, rec.normal) * distrib->SmithG1(rec.wo, h, rec.normal), //阴影-遮蔽项
                cos_theta_o = glm::dot(rec.wo, rec.normal);                                             //出射光线方向和宏观表面法线方向夹角的余弦
            auto value = Spectrum(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
            if (albedo_avg_ > 0)
                value += EvalMultipleScatter(cos_theta_i, cos_theta_o);
            if (specular_reflectance_)
                value *= specular_reflectance_->Color(rec.texcoord);
            rec.attenuation += value;
        }
        //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
        rec.attenuation *= cos_theta_i;
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    void Eval(SamplingRecord &rec) const override
    {
        Float cos_theta_o = glm::dot(rec.wo, rec.normal); //出射光线方向和宏观表面法线方向夹角的余弦
        if (cos_theta_o < 0)
        { //表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
            //又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
            //故只需确保光线出射方向和表面法线方向在介质同侧即可
            return;
        }
        //计算光线传播概率
        Float kr_i = Fresnel(rec.wi, rec.normal, eta_inv_),                  //入射菲涅尔项
            specular_sampling_weight = SpecularSamplingWeight(rec.texcoord), //抽样镜面反射的权重
            pdf_specular = kr_i * specular_sampling_weight,                  //抽样镜面反射分量的概率
            pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);   //抽样漫反射分量的概率
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec.pdf = (1.0 - pdf_specular) * PdfHemisCos(rec.wo, rec.normal);
        auto [alpha_u, alpha_v] = GetAlpha(rec.texcoord);            //景物表面沿切线方向和副切线方向的粗糙程度
        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v); //微表面分布
        Vector3 h = glm::normalize(-rec.wi + rec.wo);                //微表面法线
        Float D = distrib->Pdf(h, rec.normal);                       //微表面法线分布概率（相对于宏观表面法线）
        if (D > kEpsilonPdf)
        {
            rec.pdf += pdf_specular * D * std::abs(1.0 / (4.0 * glm::dot(rec.wo, h)));
        }
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kReflect;
        //计算光能衰减系数
        Spectrum diffuse_reflectance = diffuse_reflectance_->Color(rec.texcoord);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_);
        Float cos_theta_i = glm::dot(-rec.wi, rec.normal), //入射光线方向和宏观表面法线方向夹角的余弦
            kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_); //出射菲涅尔项
        rec.attenuation *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        if (D > kEpsilonPdf)
        {
            Float F = Fresnel(rec.wi, h, eta_inv_),                                                     //菲涅尔项
                G = distrib->SmithG1(-rec.wi, h, rec.normal) * distrib->SmithG1(rec.wo, h, rec.normal); //阴影-遮蔽项
            auto value = Spectrum(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
            if (albedo_avg_ > 0)
                value += EvalMultipleScatter(cos_theta_i, cos_theta_o);
            if (specular_reflectance_)
                value *= specular_reflectance_->Color(rec.texcoord);
            rec.attenuation += value;
        }
        //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
        rec.attenuation *= cos_theta_i;
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || Microfacet::TextureMapping() || !diffuse_reflectance_->Constant() ||
               specular_reflectance_ && !specular_reflectance_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Bsdf::Transparent(texcoord) || diffuse_reflectance_->Transparent(texcoord);
    }

private:
    ///\brief 补偿多次散射后又射出的光能
    Spectrum EvalMultipleScatter(Float cos_theta_i, Float cos_theta_o) const
    {
        Float albedo_i = GetAlbedo(std::abs(cos_theta_i)),
              albedo_o = GetAlbedo(std::abs(cos_theta_o)),
              f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
        return Spectrum(f_ms * f_add_);
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