#include "conductor.hpp"

#include "../core/ray.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

Conductor::Conductor(const std::string &id, const dvec3 &eta, const dvec3 &k, Texture *specular_reflectance)
    : Bsdf(BsdfType::kConductor, id),
      eta_(eta),
      k_(k),
      specular_reflectance_(specular_reflectance)
{
}

void Conductor::Sample(SamplingRecord *rec, Sampler* sampler) const
{
    rec->type = ScatteringType::kReflect;
    rec->pdf = 1;
    rec->wi = -Reflect(-rec->wo, rec->normal);
    rec->attenuation = FresnelConductor(rec->wi, rec->normal, eta_, k_) * glm::dot(-rec->wi, rec->normal) *
                       specular_reflectance_->color(rec->texcoord);
}

void Conductor::Eval(SamplingRecord *rec) const
{
    if (!SameDirection(rec->wo, Reflect(rec->wi, rec->normal)))
    {
        return;
    }

    rec->type = ScatteringType::kReflect;
    rec->pdf = 1;
    rec->attenuation = FresnelConductor(rec->wi, rec->normal, eta_, k_) * glm::dot(-rec->wi, rec->normal) *
                       specular_reflectance_->color(rec->texcoord);
}

bool Conductor::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || !specular_reflectance_->IsConstant();
}

NAMESPACE_END(raytracer)