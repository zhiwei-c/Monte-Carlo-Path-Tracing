#pragma once

#include "emitter.hpp"
#include "../core/intersection.hpp"

NAMESPACE_BEGIN(raytracer)

//天空
class Sky : public Emitter
{
public:
    Sky(const dvec3 &sun_direction, const dvec3 &albedo, double turbidity, double stretch, double sun_scale,
        double sky_scale, double sun_radius_scale, int resolution, bool extend);
    ~Sky();

    SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler* sampler,
                          Accelerator *accelerator) const override;
    double Pdf(const dvec3 &look_dir) const override;
    dvec3 radiance(const dvec3 &position, const dvec3 &wi) const override;

private:
    Emitter *envmap_;     //天空纹理对应的环境映射
    Texture *background_; //天空纹理
};

NAMESPACE_END(raytracer)