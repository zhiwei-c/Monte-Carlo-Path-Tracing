#include "clear_coated_conductor.hpp"

#include "rough_conductor.hpp"
#include "../core/ray.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../ndfs/ndf.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

ClearCoatedConductor::ClearCoatedConductor(const std::string &id, const dvec3 &eta, const dvec3 &k, Ndf *ndf, double clear_coat, Ndf *ndf_coat, Texture *specular_reflectance)
    : Bsdf(BsdfType::kClearCoatedConductor, id),
      nested_conductor_(new RoughConductor(id, eta, k, ndf, specular_reflectance)),
      ndf_coat_(ndf_coat),
      clear_coat_(clear_coat)
{
}

ClearCoatedConductor::~ClearCoatedConductor()
{
    delete nested_conductor_;
    nested_conductor_ = nullptr;
    delete ndf_coat_;
    ndf_coat_ = nullptr;
}

void ClearCoatedConductor::Sample(SamplingRecord *rec, Sampler *sampler) const
{
    double pdf_coat = 0,
           pdf_nested = 0;
    dvec3 attenuation_nested = dvec3(0),
          attenuation_coat = dvec3(0);

    double cos_theta_o = glm::dot(rec->wo, rec->normal); // 出射光线方向和宏观表面法线方向夹角的余弦
    double weight_coat = clear_coat_ * FresnelDielectric(-rec->wo, rec->normal, 1.0 / 1.5);
    if (sampler->Next1D() < weight_coat)
    {                                                                  // 抽样清漆层反射
        auto [alpha_u, alpha_v] = ndf_coat_->roughness(rec->texcoord); // 清漆表面沿切线方向和副切线方向的粗糙程度
        auto h = dvec3(0);                                             // 微表面法线
        double D_coat = 0.0;                                           // 微表面法线分布概率（相对于宏观表面法线）
        ndf_coat_->Sample(rec->normal, alpha_u, alpha_v, sampler->Next2D(), &h, &D_coat);
        rec->wi = -Reflect(-rec->wo, h);
        double cos_theta_i = glm::dot(-rec->wi, rec->normal); // 入射光线方向和宏观表面法线方向夹角的余弦
        if (cos_theta_i <= 0.0)
        {
            return;
        }

        const double F_coat = FresnelDielectric(rec->wi, h, 1.0 / 1.5); // 菲涅尔项
        weight_coat = clear_coat_ * F_coat;

        pdf_coat = D_coat * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));

        SamplingRecord rec_nested = *rec;
        nested_conductor_->Eval(&rec_nested);
        attenuation_nested = rec_nested.attenuation;
        pdf_nested = rec_nested.pdf;

        rec->pdf = pdf_nested * (1 - weight_coat) + weight_coat * pdf_coat;
        if (rec->pdf == 0)
        {
            return;
        }
        rec->type = ScatteringType::kReflect;

        double G_coat = ndf_coat_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
                        ndf_coat_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v); // 阴影-遮蔽项
        attenuation_coat = dvec3(F_coat * D_coat * G_coat / std::abs(4.0 * cos_theta_i * cos_theta_o)) * cos_theta_i;

        rec->attenuation = attenuation_nested * (1 - weight_coat) + clear_coat_ * attenuation_coat;
    }
    else
    { // 抽样清漆层透射
        nested_conductor_->Sample(rec, sampler);
        double cos_theta_i = glm::dot(-rec->wi, rec->normal); // 入射光线方向和宏观表面法线方向夹角的余弦
        if (cos_theta_i <= 0.0)
        {
            return;
        }

        dvec3 attenuation_nested = rec->attenuation;
        double pdf_nested = rec->pdf;
        dvec3 h = glm::normalize(-rec->wi + rec->wo); // 微表面法线

        const double F_coat = FresnelDielectric(rec->wi, h, 1.0 / 1.5); // 菲涅尔项
        double weight_coat = clear_coat_ * F_coat;

        double pdf_coat = 0;
        dvec3 attenuation_coat = dvec3(0);
        auto [alpha_u, alpha_v] = ndf_coat_->roughness(rec->texcoord);    // 景物表面沿切线方向和副切线方向的粗糙程度
        double D_coat = ndf_coat_->Pdf(h, rec->normal, alpha_u, alpha_v); // 微表面法线分布概率（相对于宏观表面法线）
        if (D_coat > 0)
        {
            pdf_coat = D_coat * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));
            double G_coat = ndf_coat_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
                            ndf_coat_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), // 阴影-遮蔽项
                cos_theta_i = glm::dot(-rec->wi, rec->normal);                             // 入射光线方向和宏观表面法线方向夹角的余弦
            attenuation_coat = dvec3(F_coat * D_coat * G_coat / std::abs(4.0 * cos_theta_i * cos_theta_o)) * cos_theta_i;
        }

        rec->pdf = pdf_nested * (1 - weight_coat) + weight_coat * pdf_coat;
        if (rec->pdf == 0)
        {
            return;
        }
        rec->type = ScatteringType::kReflect;

        rec->attenuation = attenuation_nested * (1 - weight_coat) + clear_coat_ * attenuation_coat;
    }
}

void ClearCoatedConductor::Eval(SamplingRecord *rec) const
{
    double cos_theta_o = glm::dot(rec->wo, rec->normal); // 出射光线方向和宏观表面法线方向夹角的余弦
    if (cos_theta_o <= 0.0)
    { // 表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
        // 又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
        // 故只需确保光线出射方向和表面法线方向在介质同侧即可
        return;
    }

    nested_conductor_->Eval(rec);
    dvec3 attenuation_nested = rec->attenuation;
    double pdf_nested = rec->pdf;
    dvec3 h = glm::normalize(-rec->wi + rec->wo); // 微表面法线

    const double F_coat = FresnelDielectric(rec->wi, h, 1.0 / 1.5); // 清漆层菲涅尔项
    double weight_coat = clear_coat_ * F_coat;

    double pdf_coat = 0;
    dvec3 attenuation_coat = dvec3(0);
    auto [alpha_u, alpha_v] = ndf_coat_->roughness(rec->texcoord);    // 清漆层表面沿切线方向和副切线方向的粗糙程度
    double D_coat = ndf_coat_->Pdf(h, rec->normal, alpha_u, alpha_v); // 微表面法线分布概率（相对于宏观表面法线）
    if (D_coat > 0)
    {
        pdf_coat = D_coat * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));
        double G_coat = ndf_coat_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
                        ndf_coat_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), // 阴影-遮蔽项
            cos_theta_i = glm::dot(-rec->wi, rec->normal);                             // 入射光线方向和宏观表面法线方向夹角的余弦
        attenuation_coat = dvec3(F_coat * D_coat * G_coat / std::abs(4.0 * cos_theta_i * cos_theta_o)) * cos_theta_i;
    }

    rec->pdf = pdf_nested * (1 - weight_coat) + weight_coat * pdf_coat;
    rec->attenuation = attenuation_nested * (1 - weight_coat) + clear_coat_ * attenuation_coat;

    if (rec->pdf <= kEpsilonPdf)
    {
        rec->type = ScatteringType::kNone;
    }
    else
    {
        rec->type = ScatteringType::kReflect;
    }
}

NAMESPACE_END(raytracer)