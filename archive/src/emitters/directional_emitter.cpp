#include "directional_emitter.hpp"

#include "../accelerators/accelerator.hpp"
#include "../core/ray.hpp"

NAMESPACE_BEGIN(raytracer)

DistantDirectionalEmitter::DistantDirectionalEmitter(const dvec3 &radiance, const dvec3 &direction)
    : Emitter(EmitterType::kDirectional),
      radiance_(radiance),
      direction_(direction)
{
}

SamplingRecord DistantDirectionalEmitter::Sample(const Intersection &its_shape, const dvec3 &wo, Sampler* sampler,
                                                 Accelerator *accelerator) const
{
    auto rec = SamplingRecord();
    rec.position = its_shape.position();
    rec.wo = wo;
    rec.wi = direction_;

    const Ray ray = Ray(its_shape.position(), -rec.wi);
    auto its_test = Intersection();
    if (accelerator && accelerator->Intersect(ray, sampler, &its_test))
    {
        return rec;
    }

    rec.type = ScatteringType::kScattering;
    rec.radiance = radiance_;
    rec.pdf = 1;
    return rec;
}

dvec3 DistantDirectionalEmitter::radiance(const dvec3 &position, const dvec3 &wi) const
{
    return glm::dot(wi, direction_) > 0.99 ? radiance_ : dvec3(0);
}

NAMESPACE_END(raytracer)