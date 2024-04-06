#include "rough_plastic.hpp"

#include <iostream>

#include "../core/ray.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../ndfs/ndf.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

RoughPlastic::RoughPlastic(const std::string &id, double int_ior, double ext_ior, Ndf *ndf, Texture *diffuse_reflectance,
                           Texture *specular_reflectance, bool nonlinear)
    : Bsdf(BsdfType::kRoughPlastic, id),
      ndf_(std::move(ndf)),
      eta_inv_(ext_ior / int_ior),
      fdr_(AverageFresnelDielectric(int_ior / ext_ior)),
      diffuse_reflectance_(diffuse_reflectance),
      specular_reflectance_(specular_reflectance),
      nonlinear_(nonlinear)
{
    if (UseTextureMapping())
    {
        return;
    }
    ndf_->ComputeAlbedoTable();
    if (ndf_->UseCompensation())
    {
        const double albedo_avg = ndf_->albedo_avg();
        f_add_ = Sqr(fdr_) * albedo_avg / (1.0 - fdr_ * (1.0 - albedo_avg));
    }
}

RoughPlastic::~RoughPlastic()
{
    delete ndf_;
    ndf_ = nullptr;
}

void RoughPlastic::Sample(SamplingRecord *rec, Sampler *sampler) const
{
    auto [alpha_u, alpha_v] = ndf_->roughness(rec->texcoord);             //景物表面沿切线方向和副切线方向的粗糙程度
    double kr_o = FresnelDielectric(-rec->wo, rec->normal, eta_inv_),     //出射菲涅尔项
        kr_i = kr_o,                                                      //入射菲涅尔项
        specular_sampling_weight = SpecularSamplingWeight(rec->texcoord), //抽样镜面反射的权重
        pdf_specular = kr_i * specular_sampling_weight,                   //抽样镜面反射分量的概率
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);    //抽样漫反射分量的概率=

    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    dvec3 h;  //微表面法线
    double D; //微表面法线分布概率（相对于宏观表面法线）
    if (sampler->Next1D() < pdf_specular)
    { //从镜面反射分量抽样光线方向

        //生成光线方向
        auto [alpha_u, alpha_v] = ndf_->roughness(rec->texcoord); //景物表面沿切线方向和副切线方向的粗糙程度
        ndf_->Sample(rec->normal, alpha_u, alpha_v, sampler->Next2D(), &h, &D);

        rec->wi = -Reflect(-rec->wo, h);
        rec->pdf = pdf_specular * (D * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)))) +
                   (1.0 - pdf_specular) * PdfHemisCos(rec->wo, rec->normal);
    }
    else
    { //从漫反射分量抽样光线方向
        double pdf_diffuse_local = SampleHemisCos(rec->normal, &rec->wi, sampler->Next2D());
        kr_i = FresnelDielectric(rec->wi, rec->normal, eta_inv_);
        pdf_specular = kr_i * specular_sampling_weight;
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

        h = glm::normalize(-rec->wi + rec->wo);
        D = ndf_->Pdf(h, rec->normal, alpha_u, alpha_v);
        rec->pdf = pdf_specular * (D * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)))) +
                   (1.0 - pdf_specular) * pdf_diffuse_local;
    }
    double cos_theta_i = glm::dot(-rec->wi, rec->normal); //入射光线方向和宏观表面法线方向夹角的余弦
    if (cos_theta_i <= 0.0)
    {
        return;
    }

    if (rec->pdf == 0.0)
    {
        return;
    }
    rec->type = ScatteringType::kReflect;

    //计算光能衰减系数
    dvec3 diffuse_reflectance = diffuse_reflectance_->color(rec->texcoord);
    if (nonlinear_)
    {
        rec->attenuation = diffuse_reflectance / (1.0 - diffuse_reflectance * fdr_);
    }
    else
    {
        rec->attenuation = diffuse_reflectance / (1.0 - fdr_);
    }
    rec->attenuation *= (1.0 - kr_i) * (1.0 - kr_o) * kPiRcp;
    double F = FresnelDielectric(rec->wi, h, eta_inv_), //菲涅尔项
        G = ndf_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
            ndf_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), //阴影-遮蔽项
        cos_theta_o = glm::dot(rec->wo, rec->normal);                 //入射光线方向和宏观表面法线方向夹角的余弦
    double specular = F * D * G / std::abs(4.0 * cos_theta_i * cos_theta_o);
    if (ndf_->UseCompensation())
    {
        specular += EvalMultipleScatter(cos_theta_i, cos_theta_o);
    }
    rec->attenuation += specular * specular_reflectance_->color(rec->texcoord);

    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= glm::dot(-rec->wi, rec->normal);
}

void RoughPlastic::Eval(SamplingRecord *rec) const
{
    double cos_theta_o = glm::dot(rec->wo, rec->normal); //出射光线方向和宏观表面法线方向夹角的余弦
    if (cos_theta_o <= 0.0)
    { //表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
        //又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
        //故只需确保光线出射方向和表面法线方向在介质同侧即可
        return;
    }

    //计算光线传播概率
    double kr_i = FresnelDielectric(rec->wi, rec->normal, eta_inv_),      //入射菲涅尔项
        specular_sampling_weight = SpecularSamplingWeight(rec->texcoord), //抽样镜面反射的权重
        pdf_specular = kr_i * specular_sampling_weight,                   //抽样镜面反射分量的概率
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);    //抽样漫反射分量的概率
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    rec->pdf = (1.0 - pdf_specular) * PdfHemisCos(rec->wo, rec->normal);

    dvec3 h = glm::normalize(-rec->wi + rec->wo);
    auto [alpha_u, alpha_v] = ndf_->roughness(rec->texcoord); //景物表面沿切线方向和副切线方向的粗糙程度
    double D = ndf_->Pdf(h, rec->normal, alpha_u, alpha_v);
    if (D > kEpsilonPdf)
    {
        rec->pdf += pdf_specular * D * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));
    }
    if (rec->pdf <= kEpsilonPdf)
    {
        return;
    }
    rec->type = ScatteringType::kReflect;

    //计算光能衰减系数
    dvec3 diffuse_reflectance = diffuse_reflectance_->color(rec->texcoord);
    if (nonlinear_)
    {
        rec->attenuation = diffuse_reflectance / (1.0 - diffuse_reflectance * fdr_);
    }
    else
    {
        rec->attenuation = diffuse_reflectance / (1.0 - fdr_);
    }

    double kr_o = FresnelDielectric(-rec->wo, rec->normal, eta_inv_); //出射菲涅尔项
    rec->attenuation *= (1.0 - kr_i) * (1.0 - kr_o) * kPiRcp;

    double F = FresnelDielectric(rec->wi, h, eta_inv_), //菲涅尔项
        G = ndf_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
            ndf_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), //阴影-遮蔽项
        cos_theta_i = glm::dot(-rec->wi, rec->normal);                //入射光线方向和宏观表面法线方向夹角的余弦
    double specular = F * D * G / std::abs(4.0 * cos_theta_i * cos_theta_o);
    if (ndf_->UseCompensation())
    {
        specular += EvalMultipleScatter(cos_theta_i, cos_theta_o);
    }
    rec->attenuation += specular * specular_reflectance_->color(rec->texcoord);

    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= glm::dot(-rec->wi, rec->normal);
}
double RoughPlastic::SpecularSamplingWeight(const dvec2 &texcoord) const
{
    const dvec3 kd = diffuse_reflectance_->color(texcoord),
                ks = specular_reflectance_->color(texcoord);
    const double d_sum = kd.r + kd.g + kd.b,
                 s_sum = ks.r + ks.g + ks.b;
    return s_sum / (d_sum + s_sum);
}

double RoughPlastic::EvalMultipleScatter(double cos_theta_i, double cos_theta_o) const
{
    double albedo_i = ndf_->albdo(std::abs(cos_theta_i)),
           albedo_o = ndf_->albdo(std::abs(cos_theta_o)),
           f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - ndf_->albedo_avg()));
    return f_ms * f_add_;
}

bool RoughPlastic::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || ndf_->UseTextureMapping() ||
           !diffuse_reflectance_->IsConstant() || !specular_reflectance_->IsConstant();
}

NAMESPACE_END(raytracer)