#include "plastic.hpp"

#include "../core/ray.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

Plastic::Plastic(const std::string &id, double int_ior, double ext_ior, Texture *diffuse_reflectance,
                 Texture *specular_reflectance, bool nonlinear)
    : Bsdf(BsdfType::kPlastic, id),
      eta_inv_(ext_ior / int_ior),
      fdr_(AverageFresnelDielectric(int_ior / ext_ior)),
      diffuse_reflectance_(diffuse_reflectance),
      specular_reflectance_(specular_reflectance),
      nonlinear_(nonlinear)
{
}

void Plastic::Sample(SamplingRecord *rec, Sampler* sampler) const
{
    bool add_specular = true;                                             //生成的光线方向是否在镜面反射波瓣之中
    double kr_o = FresnelDielectric(-rec->wo, rec->normal, eta_inv_),     //出射菲涅尔项
        kr_i = kr_o,                                                      //入射菲涅尔项
        specular_sampling_weight = SpecularSamplingWeight(rec->texcoord), //抽样镜面反射的权重
        pdf_specular = kr_i * specular_sampling_weight,                   //抽样镜面反射分量的概率
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);    //抽样漫反射分量的概率=

    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    if (sampler->Next1D() < pdf_specular)
    { //从镜面反射分量抽样光线方向

        //生成光线方向
        rec->wi = -Reflect(-rec->wo, rec->normal);
        rec->pdf = pdf_specular + (1.0 - pdf_specular) * PdfHemisCos(rec->wo, rec->normal);
    }
    else
    { //从漫反射分量抽样光线方向
        double pdf_diffuse_local = SampleHemisCos(rec->normal, &rec->wi, sampler->Next2D());
        kr_i = FresnelDielectric(rec->wi, rec->normal, eta_inv_);
        pdf_specular = kr_i * specular_sampling_weight;
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec->pdf = (1.0 - pdf_specular) * pdf_diffuse_local;
        if (SameDirection(rec->wi, -Reflect(-rec->wo, rec->normal)))
        {
            rec->pdf += pdf_specular;
        }
        else
        {
            add_specular = false;
        }
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
    if (add_specular)
    {
        rec->attenuation += dvec3(kr_i) * specular_reflectance_->color(rec->texcoord);
    }
    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= glm::dot(-rec->wi, rec->normal);
}

void Plastic::Eval(SamplingRecord *rec) const
{
    if (glm::dot(rec->wo, rec->normal) < 0)
    { //表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
        //又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
        //故只需确保光线出射方向和表面法线方向在介质同侧即可
        return;
    }

    //计算光线传播概率
    bool sampled_specular = false;                                        //是否抽样到了镜面反射分量
    double kr_i = FresnelDielectric(rec->wi, rec->normal, eta_inv_),      //入射菲涅尔项
        specular_sampling_weight = SpecularSamplingWeight(rec->texcoord), //抽样镜面反射的权重
        pdf_specular = kr_i * specular_sampling_weight,                   //抽样镜面反射分量的概率
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);    //抽样漫反射分量的概率
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    rec->pdf = (1.0 - pdf_specular) * PdfHemisCos(rec->wo, rec->normal);
    if (SameDirection(rec->wo, Reflect(rec->wi, rec->normal)))
    { //如果出射方向位于镜面反射波瓣之内，则再加上镜面反射成分的概率
        rec->pdf += pdf_specular;
        sampled_specular = true;
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
    if (sampled_specular)
    {
        rec->attenuation += kr_i * specular_reflectance_->color(rec->texcoord);
    }
    //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
    rec->attenuation *= glm::dot(-rec->wi, rec->normal);
}
double Plastic::SpecularSamplingWeight(const dvec2 &texcoord) const
{
    const dvec3 kd = diffuse_reflectance_->color(texcoord),
                ks = specular_reflectance_->color(texcoord);
    const double d_sum = kd.r + kd.g + kd.b,
                 s_sum = ks.r + ks.g + ks.b;
    return s_sum / (d_sum + s_sum);
}

bool Plastic::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || !diffuse_reflectance_->IsConstant() || !specular_reflectance_->IsConstant();
}

NAMESPACE_END(raytracer)