#include "triangle.hpp"

#include "../accelerators/aabb.hpp"
#include "../bsdfs/bsdf.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

Triangle::Triangle(const std::string &id, const std::vector<dvec3> &positions, const std::vector<dvec3> &normals,
                   const std::vector<dvec3> &tangents, const std::vector<dvec3> &bitangents,
                   const std::vector<dvec2> &texcoords, bool flip_normals)
    : Shape(id, ShapeType::kTriangle, flip_normals),
      positions_(positions),
      normals_(normals),
      texcoords_(texcoords),
      tangents_(tangents),
      bitangents_(bitangents),
      v0v1_(positions[1] - positions[0]),
      v0v2_(positions[2] - positions[0])
{
    area_ = glm::length(glm::cross(positions_[1] - positions_[0], positions_[2] - positions_[0])) * 0.5;
    pdf_area_ = 1.0 / area_;
    aabb_ = AABB();
    for (const dvec3 &v : positions_)
    {
        aabb_ += v;
    }
}

void Triangle::Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const
{
    dvec3 P = glm::cross(ray.dir(), v0v2_);
    double det = glm::dot(v0v1_, P);
    if (std::abs(det) < kEpsilonCompare)
    {
        return; // 丢弃与三角面片平行的光线
    }

    dvec3 T = ray.origin() - positions_[0],
          Q = glm::cross(T, v0v1_);
    const double det_inv = 1.0 / det;

    const double v = glm::dot(T, P) * det_inv;
    if (v < 0.0 || v > 1.0)
    {
        return;
    }

    const double w = glm::dot(ray.dir(), Q) * det_inv;
    if (w < 0.0 || (v + w) > 1.0)
    {
        return;
    }

    const double distance = glm::dot(v0v2_, Q) * det_inv;
    if (distance < kEpsilonDistance || its->distance() < distance || ray.t_max() < distance)
    {
        return;
    }

    const double u = 1.0 - v - w;
    dvec2 texcoord = u * texcoords_[0] + v * texcoords_[1] + w * texcoords_[2];
    if (bsdf_ != nullptr)
    {
        if (bsdf_->IsTransparent(texcoord, sampler))
        {
            return;
        }
        if (!bsdf_->IsTwosided() && (flip_normals_ && det > 0.0 || !flip_normals_ && det < 0.0))
        { // 丢弃与单面材质的三角面片交于背面的光线
            *its = Intersection(id_, distance, medium_int_, medium_ext_);
            return;
        }
    }

    dvec3 position = u * positions_[0] + v * positions_[1] + w * positions_[2];
    dvec3 normal = u * normals_[0] + v * normals_[1] + w * normals_[2];
    dvec3 tangent = u * tangents_[0] + v * tangents_[1] + w * tangents_[2];
    dvec3 bitangent = u * bitangents_[0] + v * bitangents_[1] + w * bitangents_[2];

    if (bsdf_ != nullptr)
    {
        bsdf_->ApplyBumpMapping(tangent, bitangent, texcoord, &normal);
    }

    bool inside = det < 0;
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

Intersection Triangle::SamplePoint(Sampler *sampler) const
{
    const double a = sqrt(1.0f - sampler->Next1D()),
                 v = 1.0 - a,
                 w = a * sampler->Next1D(),
                 u = 1.0 - v - w;
    dvec3 position = u * positions_[0] + v * positions_[1] + w * positions_[2],
          normal = u * normals_[0] + v * normals_[1] + w * normals_[2];
    if (flip_normals_)
    {
        normal = -normal;
    }
    return Intersection(id_, position, normal, pdf_area_, bsdf_, medium_int_, medium_ext_);
}

NAMESPACE_END(raytracer)