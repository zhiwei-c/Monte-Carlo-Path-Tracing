#pragma once

#include <vector>

#include "integrator.hpp"
#include "../core/intersection.hpp"
#include "../core/sampling_record.hpp"

NAMESPACE_BEGIN(raytracer)

//临时记录的光源路径点
struct PathVertex
{
    Intersection its;   //光源路径点的几何与光照模型信息
    SamplingRecord rec; //生成路径点时的抽样记录

    PathVertex();
    PathVertex(const Intersection &its, const SamplingRecord &rec);
};

//双向路径跟踪算法
class BdptIntegrator : public Integrator
{
public:
    BdptIntegrator(int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
                   const std::vector<Emitter *> &emitters, size_t shape_num);

    dvec3 Shade(const dvec3 &eye, const dvec3 &look_dir, Sampler* sampler) const override;

private:
    std::vector<PathVertex> CreateEmitterPath(Sampler* sampler) const;

    dvec3 ShadeIterately(Intersection its, dvec3 wo, const std::vector<PathVertex> &emitter_path, Sampler* sampler) const;
};

NAMESPACE_END(raytracer)