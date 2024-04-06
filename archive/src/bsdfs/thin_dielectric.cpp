#include "thin_dielectric.hpp"

#include "../core/ray.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../textures/texture.hpp"
#ifdef ROUGH_SMOOTH
#include "../textures/constant_texture.hpp"
#include "../ndfs/ndf.hpp"
#endif

NAMESPACE_BEGIN(raytracer)
#ifdef ROUGH_SMOOTH
static constexpr double kAlpha = 0.001;
#endif

ThinDielectric::ThinDielectric(const std::string &id, double int_ior, double ext_ior, Texture *specular_reflectance,
                               Texture *specular_transmittance)
    : Bsdf(BsdfType::kThinDielectric, id),
      eta_inv_(ext_ior / int_ior),
      specular_reflectance_(specular_reflectance),
      specular_transmittance_(specular_transmittance)
{
#ifdef ROUGH_SMOOTH
    ndf_ = new GgxNdf(nullptr, nullptr);
#endif
}
#ifdef ROUGH_SMOOTH
ThinDielectric::~ThinDielectric()
{
    delete ndf_;
    ndf_ = nullptr;
}
#endif

void ThinDielectric::Sample(SamplingRecord *rec, Sampler* sampler) const
{
    double kr = FresnelDielectric(-rec->wo, rec->normal, eta_inv_); //菲涅尔项
    if (kr < 1.0)
    {
        kr *= 2.0 / (1.0 + kr);
    }
#ifdef ROUGH_SMOOTH
    const double scale_factor = 1.2 - 0.2 * std::sqrt(std::fabs(glm::dot(-rec->wo, rec->normal)));
    const double alpha = kAlpha * scale_factor;

    auto h = dvec3(0); //微表面法线
    double D = 0.0;
    ndf_->Sample(rec->normal, alpha, alpha, sampler->Next2D(), &h, &D);
    //生成发生反射时，反射光线的方向
    dvec3 wr = Reflect(-rec->wo, h);
    double cos_theta_i = glm::dot(wr, rec->normal);
    if (cos_theta_i <= 0.0)
    {
        return;
    }
    //计算抽样微表面法线的概率
    rec->pdf = D * std::abs(1.0 / (4.0 * glm::dot(rec->wo, h)));
    if (rec->pdf == 0.0)
    {
        return;
    }
    double G = ndf_->SmithG1(wr, h, rec->normal, alpha, alpha) *
               ndf_->SmithG1(rec->wo, h, rec->normal, alpha, alpha), //阴影-遮蔽项
        cos_theta_o = glm::dot(rec->wo, rec->normal);
    if (sampler->Next1D() < kr)
    { //抽样反射光线
        rec->type = ScatteringType::kReflect;
        //生成光线方向
        rec->wi = -wr;
        //计算光线传播概率
        rec->pdf *= kr;
        //计算光能衰减系数
        rec->attenuation = dvec3(kr * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o))) *
                           specular_reflectance_->color(rec->texcoord) * cos_theta_i;
    }
    else
    { //抽样折射光线
        rec->type = ScatteringType::kTransimission;
        //生成光线方向
        rec->wi = rec->wo;
        //计算光线传播概率
        rec->pdf *= (1.0 - kr);
        //计算光能衰减系数
        rec->attenuation = dvec3((1.0 - kr) * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o))) *
                           specular_reflectance_->color(rec->texcoord) * cos_theta_i;
    }
#else
    if (sampler->Next1D() < kr)
    { //抽样反射光线
        rec->type = ScatteringType::kReflect;
        //计算光线传播概率
        rec->pdf = kr;
        //生成光线方向
        rec->wi = -Reflect(-rec->wo, rec->normal);
        //计算光能衰减系数
        rec->attenuation = dvec3(kr) * glm::dot(-rec->wi, rec->normal) * specular_reflectance_->color(rec->texcoord);
    }
    else
    { //抽样折射光线
        rec->type = ScatteringType::kTransimission;
        //生成光线方向
        rec->wi = rec->wo;
        //计算光线传播概率
        rec->pdf = 1.0 - kr;
        //计算光能衰减系数
        rec->attenuation = dvec3(1.0 - kr) * glm::dot(-rec->wi, rec->normal) * specular_transmittance_->color(rec->texcoord);
    }
#endif
}

void ThinDielectric::Eval(SamplingRecord *rec) const
{
    double kr = FresnelDielectric(rec->wi, rec->normal, eta_inv_); //菲涅尔项
    if (kr < 1.0)
    {
        kr *= 2.0 / (1.0 + kr);
    }
    if (SameDirection(rec->wo, Reflect(rec->wi, rec->normal)))
    { //处理镜面反射
        rec->pdf = kr;
        rec->type = ScatteringType::kReflect;
        rec->attenuation = dvec3(kr) * glm::dot(-rec->wi, rec->normal) * specular_reflectance_->color(rec->texcoord);
    }
    else if (SameDirection(rec->wo, rec->wi))
    { //处理折射
        rec->pdf = 1.0 - kr;
        rec->type = ScatteringType::kTransimission;
        rec->attenuation = dvec3(1.0 - kr) * glm::dot(-rec->wi, rec->normal) * specular_transmittance_->color(rec->texcoord);
    }
    else
    {
        return;
    }
}

bool ThinDielectric::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || !specular_reflectance_->IsConstant() || !specular_transmittance_->IsConstant();
}

NAMESPACE_END(raytracer)