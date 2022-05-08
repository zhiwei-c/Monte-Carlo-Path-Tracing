#include "disk.h"

NAMESPACE_BEGIN(simple_renderer)

Disk::Disk(Material *material,
           std::unique_ptr<Mat4> to_world,
           bool flip_normals)
    : Shape(ShapeType::kDisk, material, flip_normals),
      to_world_(std::move(to_world)),
      to_world_norm_(nullptr),
      to_local_(nullptr)
{
    auto center = Vector3(0, 0, 0);
    auto p1 = Vector3(0.5, 0, 0);
    auto pos_max = Vector3(-0.5, -0.5, 0);
    auto pos_min = Vector3(0.5, 0.5, 0);

    if (to_world_)
    {
        center = TransfromPt(*to_world_, center);
        p1 = TransfromPt(*to_world_, p1);
        pos_max = TransfromPt(*to_world_, pos_max);
        pos_min = TransfromPt(*to_world_, pos_min);
        to_local_.reset(new Mat4(glm::inverse(*to_world_)));
        to_world_norm_.reset(new Mat4(glm::inverse(glm::transpose(*to_world_))));
    }

    auto radius_world = glm::length(center - p1);

    area_ = kPi * radius_world * radius_world;
    pdf_area_ = 1 / area_;
    aabb_ = AABB();
    aabb_ += pos_max;
    aabb_ += pos_min;
}

void Disk::Intersect(const Ray &ray, Intersection &its) const
{
    auto ray_o = ray.origin(),
         ray_d = ray.dir();
    if (to_local_)
    {
        ray_o = TransfromPt(*to_local_, ray_o);
        ray_d = TransfromDir(*to_local_, ray_d);
    }

    auto normal = Vector3(0, 0, 1);

    auto t_z = -ray_o.z / ray_d.z;
    if (t_z < kEpsilon)
        return;

    auto pos = ray_o + t_z * ray_d;

    if (glm::length(pos) > 0.5)
        return;

    auto texcoord = Vector2(-1);
    if (material_->TextureMapping())
    {
        auto theta = static_cast<Float>(0),
             phi = static_cast<Float>(0),
             r = static_cast<Float>(0);
        CartesianToSpherical(pos, theta, phi, r);
        auto texcoord = Vector2(0);
        texcoord.x = std::min((Float)1, r);
        texcoord.y = phi * 0.5 * kPiInv;

        if (material_->Transparent(texcoord))
            return;

        if (material_->NormalPerturbing())
        {
            auto r_1 = r + kEpsilon < 1 ? r + kEpsilon : r - kEpsilon;
            auto pos1 = SphericalToCartesian(theta, phi, r_1);
            auto texcoord1 = Vector2(r_1, texcoord.y);

            auto phi2 = phi + kEpsilon < 2 * kPi ? phi + kEpsilon : phi - kEpsilon;
            auto pos2 = SphericalToCartesian(theta, phi2, r);
            auto texcoord2 = Vector2(texcoord.x, phi2 * 0.5 * kPiInv);

            auto v0v1 = pos1 - pos,
                 v0v2 = pos2 - pos;
            auto delta_uv_1 = texcoord1 - texcoord,
                 delta_uv_2 = texcoord2 - texcoord;

            auto r = 1 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
            auto tangent = glm::normalize(Vector3(delta_uv_1.y * v0v2 - delta_uv_2.y * v0v1) * r),
                 bitangent = glm::normalize(Vector3(delta_uv_2.x * v0v1 - delta_uv_1.x * v0v2) * r);

            normal = material_->PerturbNormal(normal, tangent, bitangent, texcoord);
        }
    }
    if (to_world_)
    {
        pos = TransfromPt(*to_world_, pos);
        normal = TransfromDir(*to_world_norm_, normal);
    }
    auto distance = glm::length(pos - ray.origin());
    if (distance > its.distance())
        return;

    if (!material_->Twosided() &&
        (flip_normals_ && NotSameHemis(normal, ray_d) ||
         !flip_normals_ && SameHemis(normal, ray_d)))
    {
        its = Intersection(distance);
        return;
    }

    auto inside = false;
    if (ray_o.z < 0)
    {
        normal = -normal;
        inside = !inside;
    }

    if (flip_normals_)
    {
        normal = -normal;
        inside = !inside;
    }

    its = Intersection(pos, normal, texcoord, inside, distance, material_, pdf_area_);
}

Intersection Disk::SampleP() const
{
    auto pos_xy = DiskUnifrom();
    auto pos = Vector3(pos_xy.x * 0.5, pos_xy.y * 0.5, 0);
    auto normal = Vector3(0, 0, 1);

    if (to_world_)
    {
        pos = TransfromPt(*to_world_, pos);
        normal = TransfromDir(*to_world_norm_, normal);
    }
    return Intersection(pos, normal, Vector2(-1), false, INFINITY, material_, pdf_area_);
}

NAMESPACE_END(simple_renderer)