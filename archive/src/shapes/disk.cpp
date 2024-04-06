#include "disk.hpp"

#include "../bsdfs/bsdf.hpp"
#include "../math/coordinate.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

Disk::Disk(const std::string &id, const dmat4 &to_world, bool flip_normals)
    : Shape(id, ShapeType::kDisk, flip_normals),
      to_local_(glm::inverse(to_world)),
      to_world_(to_world),
      normal_to_world_(glm::inverse(glm::transpose(to_world)))
{
    const dvec3 center = TransfromPoint(to_world, {0, 0, 0}),
                p1 = TransfromPoint(to_world, {0.5, 0, 0}),
                p2 = TransfromPoint(to_world, {-0.5, -0.5, 0}),
                p3 = TransfromPoint(to_world, {0.5, 0.5, 0});
    const double radius_world = glm::length(center - p1);
    area_ = kPi * radius_world * radius_world;
    pdf_area_ = 1.0 / area_;
    aabb_ = AABB();
    aabb_ += p2;
    aabb_ += p3;
}

void Disk::Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const
{
    const dvec3 ray_o = TransfromPoint(to_local_, ray.origin()),
                ray_d = TransfromVec(to_local_, ray.dir());

    double t_z = -ray_o.z / ray_d.z;
    if (t_z < kEpsilonDistance)
    {
        return;
    }

    dvec3 position_local = ray_o + t_z * ray_d;
    if (glm::length(position_local) > 0.5)
    {
        return;
    }

    dvec3 position = TransfromPoint(to_world_, position_local);
    double distance = glm::length(position - ray.origin());
    if (its->distance() < distance || ray.t_max() < distance)
    {
        return;
    }

    double theta = 0, phi = 0, r = 0;
    CartesianToSpherical(position_local, &theta, &phi, &r);
    dvec2 texcoord = {std::min(1.0, r), phi * 0.5 * kPiRcp};
    if (bsdf_ != nullptr)
    {
        if (bsdf_->IsTransparent(texcoord, sampler))
        {
            return;
        }
        if (!bsdf_->IsTwosided() && (flip_normals_ && ray_o.z > 0.0 || !flip_normals_ && ray_o.z < 0.0))
        {
            *its = Intersection(id_, distance, medium_int_, medium_ext_);
            return;
        }
    }

    double r1 = r + kEpsilonJitter < 1 ? r + kEpsilonJitter : r - kEpsilonJitter;
    dvec3 p1 = TransfromPoint(to_world_, SphericalToCartesian(theta, phi, r1));
    dvec2 texcoord1 = {r1, texcoord.y};

    double phi2 = phi + kEpsilonJitter < 2 * kPi ? phi + kEpsilonJitter : phi - kEpsilonJitter;
    dvec3 p2 = TransfromPoint(to_world_, SphericalToCartesian(theta, phi2, r));
    dvec2 texcoord2 = {texcoord.x, phi2 * 0.5 * kPiRcp};

    dvec3 v0v1 = p1 - position,
          v0v2 = p2 - position;
    dvec2 delta_uv_1 = texcoord1 - texcoord,
          delta_uv_2 = texcoord2 - texcoord;

    double norm = 1.0 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
    dvec3 tangent = glm::normalize(dvec3(delta_uv_1.y * v0v2 - delta_uv_2.y * v0v1) * norm),
          normal = TransfromVec(normal_to_world_, {0, 0, 1}),
          bitangent = glm::normalize(glm::cross(tangent, normal));

    if (bsdf_ != nullptr)
    {
        bsdf_->ApplyBumpMapping(tangent, bitangent, texcoord, &normal);
    }
    
    auto inside = ray_o.z < 0.0;
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

Intersection Disk::SamplePoint(Sampler *sampler) const
{

    double xi_1 = 2.0 * sampler->Next1D() - 1.0,
           xi_2 = 2.0 * sampler->Next1D() - 1.0;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
    double phi = 0, radius = 0;
    if (xi_1 == 0.0 && xi_2 == 0.0)
    {
        radius = phi = 0;
    }
    else if (Sqr(xi_1) > Sqr(xi_2))
    {
        radius = xi_1;
        phi = (kPi * 0.25) * (xi_2 / xi_1);
    }
    else
    {
        radius = xi_2;
        phi = (kPi * 0.5) - (xi_1 / xi_2) * (kPi * 0.25);
    }
    dvec2 position_local = {radius * std::cos(phi), radius * std::sin(phi)};

    dvec3 position = TransfromPoint(to_world_, {position_local.x * 0.5, position_local.y * 0.5, 0}),
          normal = TransfromVec(normal_to_world_, {0, 0, 1});
    if (flip_normals_)
    {
        normal = -normal;
    }
    return Intersection(id_, position, normal, pdf_area_, bsdf_, medium_int_, medium_ext_);
}

NAMESPACE_END(raytracer)