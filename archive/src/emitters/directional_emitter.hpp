#pragma once

#include "emitter.hpp"
#include "../core/intersection.hpp"

NAMESPACE_BEGIN(raytracer)

//遥远的方向光
class DistantDirectionalEmitter : public Emitter
{
public:
    DistantDirectionalEmitter(const dvec3 &radiance, const dvec3 &direction);

    SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler* sampler,
                          Accelerator *accelerator) const override;

    dvec3 radiance(const dvec3 &position, const dvec3 &wi) const override;

private:
    dvec3 radiance_;  //辐射亮度
    dvec3 direction_; //世界空间下的方向
};

NAMESPACE_END(raytracer)