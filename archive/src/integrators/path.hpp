#pragma once

#include "integrator.hpp"

NAMESPACE_BEGIN(raytracer)

//路径跟踪算法
class PathIntegrator : public Integrator
{
public:
    PathIntegrator(int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
                   const std::vector<Emitter *> &emitters, size_t shape_num);

    dvec3 Shade(const dvec3 &eye, const dvec3 &look_dir, Sampler* sampler) const override;
};

NAMESPACE_END(raytracer)