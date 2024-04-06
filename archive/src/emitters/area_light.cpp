#include "area_light.hpp"

#include "../accelerators/accelerator.hpp"
#include "../bsdfs/bsdf.hpp"
#include "../shapes/shape.hpp"

NAMESPACE_BEGIN(raytracer)

AreaLight::AreaLight(Bsdf *bsdf, Shape *shape)
    : Emitter(EmitterType::kArea),
      bsdf_(bsdf),
      shape_(shape)
{
}

SamplingRecord AreaLight::Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                                 Accelerator *accelerator) const
{
    auto rec = SamplingRecord();
    rec.position = its_shape.position();
    rec.wo = wo;
    if (accelerator == nullptr)
    {
        return rec;
    }

    Intersection its_emitter = shape_->SamplePoint(sampler);
    rec.distance = glm::length(its_shape.position() - its_emitter.position());
    if (rec.distance < kEpsilonCompare)
    {
        return rec;
    }

    rec.wi = glm::normalize(its_shape.position() - its_emitter.position());
    auto ray = Ray(its_shape.position(), -rec.wi);
    auto its_test = Intersection();
    if (accelerator->Intersect(ray, sampler, &its_test) && its_test.shape_id() != its_emitter.shape_id() &&
        its_test.distance() + kEpsilonDistance < rec.distance)
    {
        return rec;
    }

    double cos_theta_prime = glm::dot(rec.wi, its_emitter.normal());
    if (cos_theta_prime <= 0.0)
    {
        return rec;
    }

    rec.type = ScatteringType::kScattering;
    rec.radiance = its_emitter.radiance();
    rec.pdf = its_emitter.pdf_area() * rec.distance * rec.distance / cos_theta_prime;
    return rec;
}

Intersection AreaLight::SamplePoint(Sampler *sampler) const
{
    return shape_->SamplePoint(sampler);
}

dvec3 AreaLight::radiance(const dvec3 &position, const dvec3 &wi) const
{
    return bsdf_->radiance();
}

NAMESPACE_END(raytracer)