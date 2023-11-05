#include "instance.cuh"

#include <exception>

#include "utils.cuh"

namespace rt
{

Instance::Info::Info() : type(Instance::Type::kNone), cube{}, sphere{}, rectangle{}, meshes{} {}

Instance::Info::Info(const Instance::Info &info) : type(info.type)
{
    switch (info.type)
    {
    case Instance::Type::kNone:
        break;
    case Instance::Type::kCube:
        cube = info.cube;
        break;
    case Instance::Type::kSphere:
        sphere = info.sphere;
        break;
    case Instance::Type::kRectangle:
        rectangle = info.rectangle;
        break;
    case Instance::Type::kMeshes:
        meshes = info.meshes;
        break;
    default:
        throw std::exception("unknow instance type.");
        break;
    }
}

void Instance::Info::operator=(const Instance::Info &info)
{
    type = info.type;
    switch (info.type)
    {
    case Instance::Type::kNone:
        break;
    case Instance::Type::kCube:
        cube = info.cube;
        break;
    case Instance::Type::kSphere:
        sphere = info.sphere;
        break;
    case Instance::Type::kRectangle:
        rectangle = info.rectangle;
        break;
    case Instance::Type::kMeshes:
        meshes = info.meshes;
        break;
    default:
        throw std::exception("unknow instance type.");
        break;
    }
}

QUALIFIER_D_H Instance::Instance() : id_instance_(kInvalidId), accel_(nullptr) {}

QUALIFIER_D_H Instance::Instance(const uint32_t id_instance, BLAS *accel)
    : id_instance_(id_instance), accel_(accel)
{
}

QUALIFIER_D_H void Instance::Intersect(Ray *ray, Hit *hit) const
{
    Ray ray_local = *ray;
    Hit hit_local;
    accel_->Intersect(&ray_local, &hit_local);
    if (hit_local.valid && ray_local.t_max <= ray->t_max)
    {
        *ray = ray_local;
        *hit = hit_local;
        hit->id_instance = id_instance_;
    }
}

QUALIFIER_D_H Hit rt::Instance::Sample(const float xi_0, const float xi_1, const float xi_2) const
{
    return accel_->Sample(xi_0, xi_1, xi_2);
}

} // namespace rt