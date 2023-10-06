#pragma once

#include "integrator.hpp"

NAMESPACE_BEGIN(raytracer)

//支持体绘制的路径跟踪算法
class VolPathIntegrator : public Integrator
{
public:
    VolPathIntegrator(int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
                      const std::vector<Emitter *> &emitters, size_t shape_num);

    dvec3 Shade(const dvec3 &eye, const dvec3 &look_dir, Sampler *sampler) const override;

private:
    dvec3 SampleAreaLightsDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const override;
    dvec3 SampleOtherEmittersDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const override;
};

NAMESPACE_END(raytracer)