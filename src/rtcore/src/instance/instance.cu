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

QUALIFIER_D_H Hit Instance::Sample(const Vec3 &xi) const
{
    return blas_->Sample(xi);
}

Instance::Info Instance::Info::CreateCube(const Mat4 &to_world,
                                          const uint32_t id_bsdf)
{
    Instance::Info info;
    info.type = Instance::Type::kCube;
    info.cube.to_world = to_world;
    info.id_bsdf = id_bsdf;
    return info;
}

Instance::Info Instance::Info::CreateSphere(const float &radius,
                                            const Vec3 &center,
                                            const Mat4 &to_world,
                                            const uint32_t id_bsdf)
{
    Instance::Info info;
    info.type = Instance::Type::kSphere;
    info.sphere.radius = radius;
    info.sphere.center = center;
    info.sphere.to_world = to_world;
    info.id_bsdf = id_bsdf;
    return info;
}

Instance::Info Instance::Info::CreateRectangle(const Mat4 &to_world,
                                               const uint32_t id_bsdf)
{
    Instance::Info info;
    info.type = Instance::Type::kRectangle;
    info.rectangle.to_world = to_world;
    info.id_bsdf = id_bsdf;
    return info;
}

} // namespace csrt