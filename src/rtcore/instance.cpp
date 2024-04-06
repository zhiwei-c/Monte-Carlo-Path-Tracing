#include "csrt/rtcore/instance.hpp"

#include <exception>

#include "csrt/renderer/bsdfs/bsdf.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H Instance::Instance()
    : id_(kInvalidId), id_medium_int_(kInvalidId), id_medium_ext_(kInvalidId),
      blas_(nullptr)
{
}

QUALIFIER_D_H
Instance::Instance(const uint32_t id, const uint32_t id_medium_int,
                   const uint32_t id_medium_ext, const BLAS *blas_buffer)
    : id_(id), id_medium_int_(id_medium_int), id_medium_ext_(id_medium_ext),
      blas_(blas_buffer + id)
{
}

QUALIFIER_D_H void Instance::Intersect(Bsdf *bsdf_buffer,
                                       uint32_t *map_instance_bsdf,
                                       uint32_t *seed, Ray *ray, Hit *hit) const
{
    Bsdf *bsdf = nullptr;
    if (map_instance_bsdf[id_] != kInvalidId)
        bsdf = bsdf_buffer + map_instance_bsdf[id_];

    Ray ray_local = *ray;
    Hit hit_local;
    blas_->Intersect(bsdf, seed, &ray_local, &hit_local);
    if (hit_local.valid && ray_local.t_max <= ray->t_max)
    {
        *ray = ray_local;
        *hit = hit_local;
        hit->id_instance = id_;
        hit->id_medium_int = id_medium_int_;
        hit->id_medium_ext = id_medium_ext_;
    }
}

QUALIFIER_D_H bool Instance::IntersectAny(Bsdf *bsdf_buffer,
                                          uint32_t *map_instance_bsdf,
                                          uint32_t *seed, Ray *ray) const
{
    Bsdf *bsdf = nullptr;
    if (map_instance_bsdf[id_] != kInvalidId)
        bsdf = bsdf_buffer + map_instance_bsdf[id_];
    return blas_->IntersectAny(bsdf, seed, ray);
}

QUALIFIER_D_H Hit Instance::Sample(const float xi_0, const float xi_1,
                                   const float xi_2) const
{
    return blas_->Sample(xi_0, xi_1, xi_2);
}

} // namespace csrt