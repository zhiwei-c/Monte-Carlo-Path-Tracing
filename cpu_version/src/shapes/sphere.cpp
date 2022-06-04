#include "sphere.h"

NAMESPACE_BEGIN(raytracer)

Sphere::Sphere(Bsdf *bsdf,
               const Vector3 &center,
               Float radius,
               std::unique_ptr<Mat4> to_world,
               bool flip_normals)
    : Shape(ShapeType::kSphere, bsdf, flip_normals),
      center_(center),
      radius_(radius),
      to_world_(std::move(to_world)),
      to_world_norm_(nullptr),
      to_local_(nullptr)
{
    Vector3 center_world = center_;
    auto p1 = Vector3(center.x + radius, center.y, center.z);
    auto pos_max = Vector3(center.x + radius, center.y + radius, center.z + radius);
    auto pos_min = Vector3(center.x - radius, center.y - radius, center.z - radius);
    if (to_world_)
    {
        center_world = TransfromPt(*to_world_, center_world);
        p1 = TransfromPt(*to_world_, p1);
        pos_max = TransfromPt(*to_world_, pos_max);
        pos_min = TransfromPt(*to_world_, pos_min);
        to_local_.reset(new Mat4(glm::inverse(*to_world_)));
        to_world_norm_.reset(new Mat4(glm::inverse(glm::transpose(*to_world_))));
    }
    Float radius_world = glm::length(center_world - p1);
    area_ = 4.0 * kPi * radius_world * radius_world;
    pdf_area_ = 1.0 / area_;
    aabb_ = AABB();
    aabb_ += pos_max;
    aabb_ += pos_min;
}

void Sphere::Intersect(const Ray &ray, Intersection &its) const
{
    Vector3 ray_o = ray.origin(),
            ray_d = ray.dir();
    if (to_local_)
    {
        ray_o = TransfromPt(*to_local_, ray_o);
        ray_d = TransfromDir(*to_local_, ray_d);
    }
    ray_o -= center_;
    Float a = glm::dot(ray_d, ray_d),
          b = 2 * glm::dot(ray_d, ray_o),
          c = glm::dot(ray_o, ray_o) - radius_ * radius_,
          t_near = 0,
          t_far = 0;
    if (!SolveQuadratic<Float>(a, b, c, t_near, t_far))
        return;
    if (t_far < kEpsilon)
        return;
    ray_o += center_;
    Float t_result = 0;
    if (t_near < kEpsilon)
        t_result = t_far;
    else
        t_result = t_near;
    Vector3 pos = ray_o + t_result * ray_d;
    Vector3 normal = glm::normalize(pos - center_);
    auto texcoord = Vector2(0);
    if (bsdf_->TextureMapping())
    {
        Float theta = 0, phi = 0;
        CartesianToSpherical(normal, theta, phi);
        texcoord.x = phi * 0.5 * kPiInv;
        texcoord.y = theta * kPiInv;
        if (bsdf_->Transparent(texcoord))
            return;
        if (bsdf_->NormalPerturbing())
        {
            Float theta_1 = theta + kEpsilon < kPi ? theta + kEpsilon : theta - kEpsilon;
            Vector3 pos1 = SphericalToCartesian(theta_1, phi);
            auto texcoord1 = Vector2(texcoord.x, theta_1 * kPiInv);

            Float phi2 = phi + kEpsilon < 2 * kPi ? phi + kEpsilon : phi - kEpsilon;
            Vector3 pos2 = SphericalToCartesian(theta, phi2);
            auto texcoord2 = Vector2(phi2 * 0.5 * kPiInv, texcoord.y);

            Vector3 v0v1 = pos1 - pos,
                    v0v2 = pos2 - pos;
            Vector2 delta_uv_1 = texcoord1 - texcoord,
                    delta_uv_2 = texcoord2 - texcoord;

            Float r = 1.0 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
            Vector3 tangent = glm::normalize(Vector3(delta_uv_1.y * v0v2 - delta_uv_2.y * v0v1) * r),
                    bitangent = glm::normalize(Vector3(delta_uv_2.x * v0v1 - delta_uv_1.x * v0v2) * r);

            normal = bsdf_->PerturbNormal(normal, tangent, bitangent, texcoord);
        }
    }

    if (to_world_)
    {
        pos = TransfromPt(*to_world_, pos);
        normal = TransfromDir(*to_world_norm_, normal);
    }
    Float distance = glm::length(pos - ray.origin());
    if (distance > its.distance())
        return;

    if (!bsdf_->Twosided() &&
        (!flip_normals_ && c < 0 ||
         flip_normals_ && c > 0))
    {
        its = Intersection(distance);
        return;
    }

    auto inside = false;
    if (c < 0)
    {
        normal = -normal;
        inside = !inside;
    }
    if (flip_normals_)
    {
        normal = -normal;
        inside = !inside;
    }

    its = Intersection(pos, normal, texcoord, inside, distance, bsdf_, pdf_area_);
}

Intersection Sphere::SampleP() const
{
    Vector3 normal = SphereUniform();
    Vector3 pos = center_ + radius_ * normal;
    if (to_world_)
    {
        pos = TransfromPt(*to_world_, pos);
        normal = TransfromDir(*to_world_norm_, normal);
    }
    return Intersection(pos, normal, Vector2(-1), false, INFINITY, bsdf_, pdf_area_);
}

NAMESPACE_END(raytracer)