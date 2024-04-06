#include "point_light.hpp"

#include "../accelerators/accelerator.hpp"
#include "../core/ray.hpp"

NAMESPACE_BEGIN(raytracer)

PointLight::PointLight(const dvec3 &intensity, const dvec3 &position)
    : Emitter(EmitterType::kPoint),
      intensity_(intensity),
      position_(position)
{
}

SamplingRecord PointLight::Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                                  Accelerator *accelerator) const
{
    auto rec = SamplingRecord();
    rec.distance = glm::length(position_ - its_shape.position());
    if (rec.distance < kEpsilonDistance)
    {
        return rec;
    }

    rec.position = its_shape.position();
    rec.wo = wo;
    rec.wi = glm::normalize(its_shape.position() - position_);
    const Ray ray = Ray(its_shape.position(), -rec.wi, rec.distance);
    auto its_test = Intersection();
    if (accelerator && accelerator->Intersect(ray, sampler, &its_test) &&
        its_test.distance() + kEpsilonDistance < rec.distance)
    {
        return rec;
    }

    const double distance_rcp = 1.0 / rec.distance;
    rec.type = ScatteringType::kScattering;
    rec.radiance = intensity_ * distance_rcp * distance_rcp;
    rec.pdf = 1;
    return rec;
}

NAMESPACE_END(raytracer)