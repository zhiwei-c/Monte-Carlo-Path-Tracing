#include "instance.cuh"

#include <exception>

#include "utils.cuh"

namespace csrt
{

QUALIFIER_D_H Instance::Instance() : id_(kInvalidId), blas_(nullptr) {}

QUALIFIER_D_H Instance::Instance(const uint32_t id, const BLAS *blas_buffer)
    : id_(id), blas_(blas_buffer + id)
{
}

QUALIFIER_D_H void Instance::Intersect(Ray *ray, Hit *hit) const
{
    Ray ray_local = *ray;
    Hit hit_local;
    blas_->Intersect(&ray_local, &hit_local);
    if (hit_local.valid && ray_local.t_max <= ray->t_max)
    {
        *ray = ray_local;
        *hit = hit_local;
        hit->id_instance = id_;
    }
}

QUALIFIER_D_H Hit Instance::Sample(const float xi_0, const float xi_1,
                                   const float xi_2) const
{
    return blas_->Sample(xi_0, xi_1, xi_2);
}

} // namespace csrt