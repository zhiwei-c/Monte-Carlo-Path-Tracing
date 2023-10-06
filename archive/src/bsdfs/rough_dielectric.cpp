#include "rough_dielectric.hpp"

#include "../core/ray.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../ndfs/ndf.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

RoughDielectric::RoughDielectric(const std::string &id, double int_ior, double ext_ior, Ndf *ndf, Texture *specular_reflectance,
                                 Texture *specular_transmittance)
    : Bsdf(BsdfType::kRoughDielectric, id),
      ndf_(ndf),
      eta_(int_ior / ext_ior),
      eta_inv_(ext_ior / int_ior),
      specular_reflectance_(specular_reflectance),
      specular_transmittance_(specular_transmittance)
{
    if (UseTextureMapping())
    {
        return;
    }
    ndf_->ComputeAlbedoTable();
    if (ndf_->UseCompensation())
    {
        const double F_avg = AverageFresnelDielectric(eta_),
                     F_avg_inv = AverageFresnelDielectric(eta_inv_),
                     albedo_avg = ndf_->albedo_avg();

        f_add_ = F_avg * albedo_avg / (1.0 - F_avg * (1.0 - albedo_avg));
        ratio_t_ = (1.0 - F_avg) * (1.0 - F_avg_inv) * Sqr(eta_) / ((1.0 - F_avg) + (1.0 - F_avg_inv) * Sqr(eta_));

        f_add_inv_ = F_avg_inv * albedo_avg / (1.0 - F_avg_inv * (1.0 - albedo_avg));
        ratio_t_inv_ = (1.0 - F_avg_inv) * (1.0 - F_avg) * Sqr(eta_inv_) / ((1.0 - F_avg_inv) + (1.0 - F_avg) * Sqr(eta_inv_));
    }
}

RoughDielectric::~RoughDielectric()
{
    delete ndf_;
    ndf_ = nullptr;
}

void RoughDielectric::Sample(SamplingRecord *rec, Sampler* sampler) const
{
    double eta = rec->inside ? eta_inv_ : eta_,              //相对折射率，即光线透射侧介质折射率与入射侧介质折射率之比
        eta_inv = rec->inside ? eta_ : eta_inv_,             //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        ratio_t = rec->inside ? ratio_t_inv_ : ratio_t_,     //补偿多次散射后出射光能中折射的比例
        ratio_t_inv = rec->inside ? ratio_t_ : ratio_t_inv_; //光线逆向传播时，补偿多次散射后出射光能中折射的比例

    auto [alpha_u, alpha_v] = ndf_->roughness(rec->texcoord); //景物表面沿切线方向和副切线方向的粗糙程度
    double scale_factor = 1.2 - 0.2 * std::sqrt(std::fabs(glm::dot(-rec->wo, rec->normal)));
    alpha_u *= scale_factor, alpha_v *= scale_factor;

    auto h = dvec3(0); //微表面法线
    double D = 0.0;    //微表面法线分布概率（相对于宏观表面法线）

    ndf_->Sample(rec->normal, alpha_u, alpha_v, sampler->Next2D(), &h, &D);

    double F = FresnelDielectric(-rec->wo, h, eta_inv), //菲涅尔项
        cos_theta_i = 0;                                //入射光线方向和宏观表面法线方向夹角的余弦
    if (sampler->Next1D() < F)
    { //抽样反射光线
        //生成光线方向
        rec->wi = -Reflect(-rec->wo, h);
        cos_theta_i = glm::dot(-rec->wi, rec->normal);
        if (cos_theta_i <= 0.0)
        {
            return;
        }
        //计算光线传播概率
        rec->pdf = F * D * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));
        if (rec->pdf == 0.0)
        {
            return;
        }
        rec->type = ScatteringType::kReflect;
        //计算光能衰减系数
        double G = ndf_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
                   ndf_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), //阴影-遮蔽项
            cos_theta_o = glm::dot(rec->wo, rec->normal);                    //出射光线方向和宏观表面法线方向夹角的余弦
        rec->attenuation = dvec3(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
        if (ndf_->UseCompensation())
        { //补偿多次散射后又射出的光能
            const double weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec->inside);
            rec->attenuation += dvec3(weight_loss);
        }
        rec->attenuation *= specular_reflectance_->color(rec->texcoord);
    }
    else
    { //抽样折射光线
        //生成光线方向
        rec->wi = -Refract(-rec->wo, h, eta_inv);
        cos_theta_i = glm::dot(rec->wi, rec->normal); //入射光线方向和宏观表面法线方向夹角的余弦
        if (cos_theta_i <= 0.0)
        {
            return;
        }
        { //光线折射时穿过了介质，为了使得光线入射方向和表面法线方向夹角的余弦仍小于零，需做一些相应处理
            rec->normal = -rec->normal;
            rec->inside = !rec->inside;
            h = -h;
            std::swap(eta_inv, eta);
            std::swap(ratio_t, ratio_t_inv);
        }
        //计算光线传播概率
        F = FresnelDielectric(rec->wi, h, eta_inv);
        double cos_i_h = glm::dot(-rec->wi, h), //入射光线方向和微表面法线方向夹角的余弦
            cos_o_h = glm::dot(rec->wo, h);     //出射光线方向和微表面法线方向夹角的余弦
        rec->pdf = (1.0 - F) * D * std::abs(cos_o_h / Sqr(eta_inv * cos_i_h + cos_o_h));
        if (rec->pdf == 0.0)
        {
            return;
        }
        rec->type = ScatteringType::kTransimission;

        //计算光能衰减系数
        double G = ndf_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
                   ndf_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v), //阴影-遮蔽项
            cos_theta_o = glm::dot(rec->wo, rec->normal);                    //出射光线方向和宏观表面法线方向夹角的余弦
        rec->attenuation = dvec3(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
                                          (cos_theta_i * cos_theta_o * Sqr(eta_inv * cos_i_h + cos_o_h))));
        if (ndf_->UseCompensation())
        { //补偿多次散射后又射出的光能
            const double weight_loss = ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec->inside);
            rec->attenuation += dvec3(weight_loss);
        }
        rec->attenuation *= specular_transmittance_->color(rec->texcoord);
        //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);
    }
    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= cos_theta_i;
}

void RoughDielectric::Eval(SamplingRecord *rec) const
{
    double eta = rec->inside ? eta_inv_ : eta_,
           eta_inv = rec->inside ? eta_ : eta_inv_,   //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        cos_theta_o = glm::dot(rec->wo, rec->normal); //出射光线方向和宏观表面法线方向夹角的余弦
    auto h = dvec3(0);                                //微表面法线
    bool relfect = cos_theta_o > 0.0;                 //出射光线是否是反射光线
    if (relfect)
    {
        h = glm::normalize(-rec->wi + rec->wo);
    }
    else
    {
        h = glm::normalize(-eta_inv * rec->wi + rec->wo);
        if (glm::dot(h, rec->normal) <= 0.0)
        {
            h = -h;
        }
    }
    //计算光线传播概率
    auto [alpha_u, alpha_v] = ndf_->roughness(rec->texcoord); //景物表面沿切线方向和副切线方向的粗糙程度
    double D = ndf_->Pdf(h, rec->normal, alpha_u, alpha_v),  //微表面法线分布概率（相对于宏观表面法线）
        F = FresnelDielectric(rec->wi, h, eta_inv),          //菲涅尔项
        cos_i_h = glm::dot(-rec->wi, h),                     //入射光线方向和微表面法线方向夹角的余弦
        cos_o_h = glm::dot(rec->wo, h);                      //出射光线方向和微表面法线方向夹角的余弦
    if (D <= kEpsilonPdf)
    {
        return;
    }
    rec->pdf = relfect ? F * D * std::abs(1.0 / (4.0 * cos_o_h))
                       : (1.0 - F) * D * std::abs(cos_o_h / Sqr(eta_inv * cos_i_h + cos_o_h));
    if (rec->pdf <= kEpsilonPdf)
    {
        return;
    }

    rec->type = relfect ? ScatteringType::kReflect : ScatteringType::kTransimission;

    //计算光能衰减系数
    double ratio_t = rec->inside ? ratio_t_inv_ : ratio_t_, //补偿多次散射后出射光能中折射的比例
        cos_theta_i = glm::dot(-rec->wi, rec->normal),      //入射光线方向和宏观表面法线方向夹角的余弦
        G = ndf_->SmithG1(-rec->wi, h, rec->normal, alpha_u, alpha_v) *
            ndf_->SmithG1(rec->wo, h, rec->normal, alpha_u, alpha_v); //阴影-遮蔽项
    if (relfect)
    {
        rec->attenuation = dvec3(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
        if (ndf_->UseCompensation())
        {
            const double weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec->inside);
            rec->attenuation += dvec3(weight_loss);
        }
        rec->attenuation *= specular_reflectance_->color(rec->texcoord);
    }
    else
    {
        rec->attenuation = dvec3(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
                                          (cos_theta_i * cos_theta_o * Sqr(eta_inv * cos_i_h + cos_o_h))));
        if (ndf_->UseCompensation())
        {
            const double weight_loss = ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec->inside);
            rec->attenuation += dvec3(weight_loss);
        }
        rec->attenuation *= specular_transmittance_->color(rec->texcoord);
        //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);
    }
    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= cos_theta_i;
}

double RoughDielectric::EvalMultipleScatter(double cos_theta_i, double cos_theta_o, bool inside) const
{
    double f_add = inside ? f_add_inv_ : f_add_,
           albedo_i = ndf_->albdo(std::abs(cos_theta_i)),
           albedo_o = ndf_->albdo(std::abs(cos_theta_o)),
           f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - ndf_->albedo_avg()));
    return f_ms * f_add;
}

bool RoughDielectric::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || ndf_->UseTextureMapping() ||
           !specular_reflectance_->IsConstant() || !specular_transmittance_->IsConstant();
}

NAMESPACE_END(raytracer)