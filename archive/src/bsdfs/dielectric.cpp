#include "dielectric.hpp"

#include "../core/ray.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

Dielectric::Dielectric(const std::string &id, double int_ior, double ext_ior, Texture *specular_reflectance,
                       Texture *specular_transmittance)
    : Bsdf(BsdfType::kDielectric, id),
      eta_(int_ior / ext_ior),
      eta_inv_(ext_ior / int_ior),
      specular_reflectance_(specular_reflectance),
      specular_transmittance_(specular_transmittance)
{
}

void Dielectric::Sample(SamplingRecord *rec, Sampler* sampler) const
{
    double eta = rec->inside ? eta_inv_ : eta_,                 //相对折射率，即光线透射侧介质折射率与入透射侧介质折射率之比
        eta_inv = rec->inside ? eta_ : eta_inv_,                //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        kr = FresnelDielectric(-rec->wo, rec->normal, eta_inv); //菲涅尔项
    if (sampler->Next1D() < kr)
    { //抽样反射光线
        //计算光线传播概率
        rec->pdf = kr;
        rec->type = ScatteringType::kReflect;
        //生成光线方向
        rec->wi = -Reflect(-rec->wo, rec->normal);
        //计算光能衰减系数
        rec->attenuation = dvec3(kr) * glm::dot(-rec->wi, rec->normal) * specular_reflectance_->color(rec->texcoord);
    }
    else
    { //抽样折射光线
        //生成光线方向
        rec->wi = -Refract(-rec->wo, rec->normal, eta_inv);
        { //光线折射时穿过了介质，为了使实际的入射光线方向和表面法线方向夹角的余弦仍小于零，需做一些相应处理
            rec->normal = -rec->normal;
            rec->inside = !rec->inside;
            std::swap(eta_inv, eta);
        }
        kr = FresnelDielectric(rec->wi, rec->normal, eta_inv);
        //计算光线传播概率
        rec->pdf = 1.0 - kr;
        if (rec->pdf == 0.0)
        {
            return;
        }
        rec->type = ScatteringType::kTransimission;
        //计算光能衰减系数
        rec->attenuation = dvec3(1.0 - kr) * glm::dot(-rec->wi, rec->normal) * specular_transmittance_->color(rec->texcoord);
        //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);
    }
}

void Dielectric::Eval(SamplingRecord *rec) const
{
    double eta = rec->inside ? eta_inv_ : eta_,
           eta_inv = rec->inside ? eta_ : eta_inv_,            //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        kr = FresnelDielectric(rec->wi, rec->normal, eta_inv); //菲涅尔项

    if (SameDirection(rec->wo, Reflect(rec->wi, rec->normal)))
    { //处理镜面反射
        rec->pdf = kr;
        rec->type = ScatteringType::kReflect;
        rec->attenuation = dvec3(kr) * glm::dot(-rec->wi, rec->normal) * specular_reflectance_->color(rec->texcoord);
    }
    else if (SameDirection(rec->wo, Refract(rec->wi, rec->normal, eta_inv)))
    { //处理折射
        rec->pdf = 1.0 - kr;
        rec->type = ScatteringType::kTransimission;
        rec->attenuation = dvec3(1.0 - kr) * glm::dot(-rec->wi, rec->normal) * specular_transmittance_->color(rec->texcoord);
        //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);
    }
    else
    {
        return;
    }
}

bool Dielectric::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || !specular_reflectance_->IsConstant() || !specular_transmittance_->IsConstant();
}

NAMESPACE_END(raytracer)