#include "cylinder.hpp"

#include <glm/gtx/transform.hpp>

#include "../accelerators/aabb.hpp"
#include "../bsdfs/bsdf.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

Cylinder::Cylinder(const std::string &id, const dvec3 &p0, const dvec3 &p1, double radius, const dmat4 &to_world,
                   bool flip_normals)
    : Shape(id, ShapeType::kCylinder, flip_normals)
{
    dvec3 generatrix = glm::normalize(p1 - p0);
    double length = glm::length(p1 - p0);
    to_world_ = to_world * glm::translate(p0) * ToWorld(generatrix) * glm::scale(dvec3{radius, radius, length});

    radius_ = glm::length(TransfromPoint(to_world_, dvec3{1, 0, 0}) - TransfromPoint(to_world_, dvec3{0, 0, 0}));
    length_ = glm::length(TransfromPoint(to_world_, dvec3{0, 0, 1}) - TransfromPoint(to_world_, dvec3{0, 0, 0}));
    to_world_ = to_world_ * glm::scale(dvec3{1.0 / radius_, 1.0 / radius_, 1.0 / length_});

    noraml_to_world_ = glm::inverse(glm::transpose(to_world_));
    to_local_ = glm::inverse(to_world_);

    area_ = 2.0 * kPi * radius_ * radius_;
    pdf_area_ = 1.0 / area_;

    dvec3 range;
    dvec3 x1 = TransfromPoint(to_world_, dvec3{radius_, 0, 0}),
          x2 = TransfromPoint(to_world_, dvec3{0, radius_, 0});
    for (int dim = 0; dim < 3; ++dim)
    {
        range[dim] = std::sqrt(x1[dim] * x1[dim] + x2[dim] * x2[dim]);
    }
    dvec3 y0 = TransfromPoint(to_world_, dvec3{0, 0, 0}),
          y1 = TransfromPoint(to_world_, dvec3{0, 0, length_});
    aabb_ = AABB();
    aabb_ += (y0 - range);
    aabb_ += (y1 - range);
    aabb_ += (y0 + range);
    aabb_ += (y1 + range);
}

void Cylinder::Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const
{
    const dvec3 ray_o = TransfromPoint(to_local_, ray.origin()),
                ray_d = TransfromVec(to_local_, ray.dir());

    const double a = Sqr(ray_d.x) + Sqr(ray_d.y),
                 b = 2.0 * (ray_d.x * ray_o.x + ray_d.y * ray_o.y),
                 c = Sqr(ray_o.x) + Sqr(ray_o.y) - Sqr(radius_);
    double t_near = 0.0, t_far = 0.0;
    if (!SolveQuadratic(a, b, c, &t_near, &t_far))
    {
        return;
    }
    if (t_far < kEpsilonDistance)
    {
        return;
    }

    const double z_near = ray_o.z + ray_d.z * t_near,
                 z_far = ray_o.z + ray_d.z * t_far;
    double t = 0.0;

    if (kEpsilonDistance < t_near && 0.0 <= z_near && z_near <= length_)
    {
        t = t_near;
    }
    else if (0.0 <= z_far && z_far <= length_)
    {
        t = t_far;
    }
    else
    {
        return;
    }

    if (bsdf_ != nullptr && bsdf_->IsTransparent(dvec2{0, 0}, sampler))
    {
        return;
    }

    dvec3 position_local = ray_o + t * ray_d,
          normal_local = glm::normalize(dvec3{position_local.x, position_local.y, 0.0}),
          position = TransfromPoint(to_world_, position_local),
          normal = TransfromVec(noraml_to_world_, normal_local),
          tangent = glm::normalize(TransfromPoint(to_world_, position_local + dvec3{0, 0, 1}) - position),
          bitangent = glm::normalize(glm::cross(tangent, normal));
    const double distance = glm::length(position - ray.origin());

    if (bsdf_ != nullptr && !bsdf_->IsTwosided() &&
        (!flip_normals_ && (c < 0.0 || kEpsilonDistance < t_near && (z_near < 0.0 || z_near > length_)) ||
         flip_normals_ && c > 0.0))
    {
        *its = Intersection(id_, distance, medium_int_, medium_ext_);
        return;
    }

    bool inside = c < 0.0;
    if (flip_normals_)
    {
        inside = !inside;
    }
    if (inside)
    {
        normal = -normal;
        tangent = -tangent;
        bitangent = -bitangent;
    }

    *its = Intersection(id_, inside, position, normal, tangent, bitangent, dvec2{0, 0}, distance, pdf_area_, bsdf_,
                        medium_int_, medium_ext_);
}

Intersection Cylinder::SamplePoint(Sampler *sampler) const
{
    double phi = 2.0 * kPi * sampler->Next1D(),
           z = length_ * sampler->Next1D(),
           cos_phi = std::cos(phi),
           sin_phi = std::sin(phi);
    dvec3 position_local = {cos_phi * radius_, sin_phi * radius_, z},
          normal_local = {cos_phi, sin_phi, 0.0};
    if (flip_normals_)
    {
        normal_local = -normal_local;
    }
    dvec3 position = TransfromPoint(to_world_, position_local),
          normal = TransfromVec(noraml_to_world_, normal_local);
    return Intersection(id_, position, normal, pdf_area_, bsdf_, medium_int_, medium_ext_);
}

NAMESPACE_END(raytracer)