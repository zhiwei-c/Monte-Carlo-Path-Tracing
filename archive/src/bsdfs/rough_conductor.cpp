#include "rough_conductor.hpp"

#include "../core/ray.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../ndfs/ndf.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

RoughConductor::RoughConductor(const std::string &id, const dvec3 &eta, const dvec3 &k, Ndf *ndf, Texture *specular_reflectance)
    : Bsdf(BsdfType::kRoughConductor, id),
      ndf_(ndf),
      eta_(eta),
      k_(k),
      specular_reflectance_(specular_reflectance)
{
    if (UseTextureMapping())
    {
        return;
    }
    ndf_->ComputeAlbedoTable();
    if (ndf_->UseCompensation())
    {
        dvec3 F_avg = AverageFresnelConductor(eta, k);
        const double albedo_avg = ndf_->albedo_avg();
        f_add_ = Sqr(F_avg) * albedo_avg / (dvec3(1) - F_avg * (1.0 - albedo_avg));
    }
}

RoughConductor::~RoughConductor()
{
    delete ndf_;
    ndf_ = nullptr;
}

void RoughConductor::Sample(SamplingRecord *rec, Sampler* sampler) const
{
    auto [alpha_u, alpha_v] = ndf_->roughness(rec->texcoord); //景物表面沿切线方向和副切线方向的粗糙程度
    auto h = dvec3(0);                                       //微表面法线
    double D = 0.0;                                          //微表面法线分布概率（相对于宏观表面法线）
    ndf_->Sample(rec->normal, alpha_u, alpha_v, sampler->Next2D(), &h, &D);

    //计算光线传播概率
    rec->pdf = D * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));
    if (rec->pdf == 0.0)
    {
        return;
    }

    rec->wi = -Reflect(-rec->wo, h);
    double cos_theta_i = glm::dot(-rec->wi, rec->normal); //入射光线方向和宏观表面法线方向夹角的余弦
    if (cos_theta_i <= 0.0)
    {
        return;
    }

    rec->type = ScatteringType::kReflect;
    dvec3 F = FresnelConductor(rec->wi, h, eta_, k_); //菲涅尔项
    double G = ndf_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
               ndf_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), //阴影-遮蔽项
        cos_theta_o = glm::dot(rec->wo, rec->normal);                    //出射光线方向和宏观表面法线方向夹角的余弦

    rec->attenuation = F * D * G / std::abs(4.0 * cos_theta_i * cos_theta_o);
    if (ndf_->UseCompensation())
    {
        rec->attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
    }
    rec->attenuation *= specular_reflectance_->color(rec->texcoord);
    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= cos_theta_i;
}

void RoughConductor::Eval(SamplingRecord *rec) const
{
    double cos_theta_o = glm::dot(rec->wo, rec->normal); //出射光线方向和宏观表面法线方向夹角的余弦
    if (cos_theta_o <= 0.0)
    { //表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
        //又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
        //故只需确保光线出射方向和表面法线方向在介质同侧即可
        return;
    }

    //计算光线传播概率
    dvec3 h = glm::normalize(-rec->wi + rec->wo);            //微表面法线
    auto [alpha_u, alpha_v] = ndf_->roughness(rec->texcoord); //景物表面沿切线方向和副切线方向的粗糙程度
    double D = ndf_->Pdf(h, rec->normal, alpha_u, alpha_v);  //微表面法线分布概率（相对于宏观表面法线）
    if (D <= kEpsilonPdf)
    {
        return;
    }
    rec->pdf = D * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));
    if (rec->pdf <= kEpsilonPdf)
    {
        return;
    }
    rec->type = ScatteringType::kReflect;

    //计算光能衰减系数
    dvec3 F = FresnelConductor(rec->wi, h, eta_, k_); //菲涅尔项
    double G = ndf_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
               ndf_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), //阴影-遮蔽项
        cos_theta_i = glm::dot(-rec->wi, rec->normal);                   //入射光线方向和宏观表面法线方向夹角的余弦
    rec->attenuation = F * D * G / std::abs(4.0 * cos_theta_i * cos_theta_o);
    if (ndf_->UseCompensation())
    {
        rec->attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
    }
    rec->attenuation *= specular_reflectance_->color(rec->texcoord);
    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= cos_theta_i;
}

dvec3 RoughConductor::EvalMultipleScatter(double cos_theta_i, double cos_theta_o) const
{
    double albedo_i = ndf_->albdo(std::abs(cos_theta_i)),
           albedo_o = ndf_->albdo(std::abs(cos_theta_o)),
           f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - ndf_->albedo_avg()));
    return f_ms * f_add_;
}

bool RoughConductor::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || ndf_->UseTextureMapping() ||
           !specular_reflectance_->IsConstant();
}

NAMESPACE_END(raytracer)