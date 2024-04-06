#pragma once

#include "emitter.hpp"

NAMESPACE_BEGIN(raytracer)

//面光源
class AreaLight : public Emitter
{
public:
    AreaLight(Bsdf *bsdf, Shape *shape);

    SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler* sampler,
                          Accelerator *accelerator) const override;
    Intersection SamplePoint(Sampler* sampler) const override;

    dvec3 radiance(const dvec3 &position, const dvec3 &wi) const override;

private:
    Bsdf *bsdf_;   //面光源材质
    Shape *shape_; //面光源的几何形状
};

NAMESPACE_END(raytracer)