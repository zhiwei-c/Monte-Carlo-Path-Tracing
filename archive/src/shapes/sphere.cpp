#include "sphere.hpp"

#include "../accelerators/aabb.hpp"
#include "../bsdfs/bsdf.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

Sphere::Sphere(const std::string &id, const dvec3 &center, double radius, const dmat4 &to_world, bool flip_normals)
    : Shape(id, ShapeType::kSphere, flip_normals),
      center_(center),
      radius_(radius),
      to_world_(to_world),
      noraml_to_world_(glm::inverse(glm::transpose(to_world))),
      to_local_(glm::inverse(to_world))
{
    dvec3 p = {center.x + radius, center.y, center.z},
          p_max = {center.x + radius, center.y + radius, center.z + radius},
          p_min = {center.x - radius, center.y - radius, center.z - radius},
          center_world = TransfromPoint(to_world, center);
    p = TransfromPoint(to_world, p);
    p_max = TransfromPoint(to_world, p_max);
    p_min = TransfromPoint(to_world, p_min);
    const double radius_world = glm::length(center_world - p);
    area_ = (4.0 * kPi) * (radius_world * radius_world);
    pdf_area_ = 1.0 / area_;
    aabb_ = AABB();
    aabb_ += p_max;
    aabb_ += p_min;
}

void Sphere::Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const
{
    const dvec3 ray_o = TransfromPoint(to_local_, ray.origin()) - center_,
                ray_d = TransfromVec(to_local_, ray.dir());

    const double a = glm::dot(ray_d, ray_d),
           b = 2.0 * glm::dot(ray_d, ray_o),
                 c = glm::dot(ray_o, ray_o) - radius_ * radius_;
    double t_near = 0.0, t_far = 0.0;
    if (!SolveQuadratic(a, b, c, &t_near, &t_far))
    {
        return;
    }
    if (t_far < kEpsilonDistance)
    {
        return;
    }
    double t = t_near < kEpsilonDistance ? t_far : t_near;

    dvec3 position_local = ray_o + t * ray_d + center_,
          normal_local = glm::normalize(ray_o + t * ray_d);
    double theta, phi;
    CartesianToSpherical(normal_local, &theta, &phi);
    dvec2 texcoord = {phi * 0.5 * kPiRcp, theta * kPiRcp};
    if (bsdf_ != nullptr && bsdf_->IsTransparent(texcoord, sampler))
    {
        return;
    }

    dvec3 position = TransfromPoint(to_world_, position_local),
          normal = TransfromVec(noraml_to_world_, normal_local);

    const double distance = glm::length(position - ray.origin());
    if (its->distance() < distance || ray.t_max() < distance)
    {
        return;
    }
    else if (bsdf_ != nullptr && !bsdf_->IsTwosided() && (!flip_normals_ && c < 0.0 || flip_normals_ && c > 0.0))
    {
        *its = Intersection(id_, distance, medium_int_, medium_ext_);
        return;
    }

    double theta_1 = theta + kEpsilonJitter < 0.5 * kPi ? (theta + kEpsilonJitter) : (theta - kEpsilonJitter),
           phi_2 = phi + kEpsilonJitter < 2 * kPi ? phi + kEpsilonJitter : phi - kEpsilonJitter;
    dvec3 p1 = TransfromPoint(to_world_, SphericalToCartesian(theta_1, phi)),
          p2 = TransfromPoint(to_world_, SphericalToCartesian(theta, phi_2)),
          v0v1 = p1 - position,
          v0v2 = p2 - position;
    dvec2 texcoord_1 = {texcoord.x, theta_1 * kPiRcp},
          texcoord_2 = {phi_2 * 0.5 * kPiRcp, texcoord.y},
          delta_uv_1 = texcoord_1 - texcoord,
          delta_uv_2 = texcoord_2 - texcoord;
    double r = 1.0 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
    dvec3 tangent = glm::normalize(r * dvec3{delta_uv_1.y * v0v2 - delta_uv_2.y * v0v1}),
          bitangent = glm::normalize(glm::cross(tangent, normal));

    if (bsdf_ != nullptr)
    {
        bsdf_->ApplyBumpMapping(tangent, bitangent, texcoord, &normal);
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

    *its = Intersection(id_, inside, position, normal, tangent, bitangent, texcoord, distance, pdf_area_, bsdf_,
                        medium_int_, medium_ext_);
}

Intersection Sphere::SamplePoint(Sampler *sampler) const
{
    const dvec3 dir = SampleSphereUniform(sampler->Next2D());
    dvec3 position = TransfromPoint(to_world_, center_ + radius_ * dir),
          normal = TransfromVec(noraml_to_world_, dir);
    if (flip_normals_)
    {
        normal = -normal;
    }
    return Intersection(id_, position, normal, pdf_area_, bsdf_, medium_int_, medium_ext_);
}

NAMESPACE_END(raytracer)