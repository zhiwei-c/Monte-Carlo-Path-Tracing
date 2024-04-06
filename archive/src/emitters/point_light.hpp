#pragma once

#include "emitter.hpp"
#include "../core/intersection.hpp"

NAMESPACE_BEGIN(raytracer)

//点光源
class PointLight : public Emitter
{
public:
    PointLight(const dvec3 &intensity, const dvec3 &position);

    SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler* sampler,
                          Accelerator *accelerator) const override;

private:
    dvec3 intensity_; //辐射强度
    dvec3 position_;  //世界空间下的位置
};

NAMESPACE_END(raytracer)