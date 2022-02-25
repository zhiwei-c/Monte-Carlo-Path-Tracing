#pragma once

#include "../shape.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

class Sphere : public Shape
{
public:
    /**
     * \brief 球
     * \param material 材质
     * \param center 球心
     * \param radius 半径
     * \param to_world 从局部坐标系到世界坐标系的变换矩阵
     * \param flip_normals 法线方向是否翻转
     */
    Sphere(Material *material, Vector3 center, Float radius, std::unique_ptr<Mat4> to_world, bool flip_normals)
        : Shape(ShapeType::kSphere, material, flip_normals), center_(center), radius_(radius), material_(material), to_world_(std::move(to_world)), to_world_norm_(nullptr), to_local_(nullptr)
    {
        auto center_world = center_;
        auto p1 = center_ = Vector3(center.x + radius, center.y, center.z);

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
        auto radius_world = glm::length(center_world - p1);
        area_ = 4 * kPi * radius_world * radius_world;
        area_inv_ = 1 / area_;
        aabb_ = AABB();
        aabb_ += pos_max;
        aabb_ += pos_min;
    }

    Intersection Intersect(const Ray &ray) const override
    {
        auto ray_o = ray.origin(),
             ray_d = ray.dir();
        if (to_local_)
        {
            ray_o = TransfromPt(*to_local_, ray_o);
            ray_d = TransfromDir(*to_local_, ray_d);
        }
        ray_o -= center_;
        auto a = glm::dot(ray_d, ray_d);
        auto b = 2 * glm::dot(ray_d, ray_o);
        auto c = glm::dot(ray_o, ray_o) - radius_ * radius_;

        Float t_near, t_far;
        if (!SolveQuadratic<Float>(a, b, c, t_near, t_far))
            return Intersection();

        if (t_far < kEpsilon)
            return Intersection();

        Float t_result;
        if (!material_->twosided())
        {
            if (flip_normals_)
            {
                if (t_near > kEpsilon)
                    return Intersection();
                else
                    t_result = t_far;
            }
            else
            {
                if (t_near < kEpsilon)
                    return Intersection();
                else
                    t_result = t_near;
            }
        }
        else
        {
            if (t_near < kEpsilon)
                t_result = t_far;
            else
                t_result = t_near;
        }

        auto pos = ray_o + t_result * ray_d;
        auto normal = glm::normalize(pos - center_);

        auto texcoord = Vector2(-1);
        if (this->material_->NormalPerturbing() ||
            this->material_->OpacityMapping() ||
            this->material_->TextureMapping())
        {
            Float theta, phi;
            CartesianToSpherical(normal, theta, phi);

            texcoord.x = phi * 0.5 * kPiInv;
            texcoord.y = theta * kPiInv;

            if (this->material_->Transparent(texcoord))
                return Intersection();

            if (this->material_->NormalPerturbing())
            {
                auto theta_1 = theta + kEpsilon < kPi ? theta + kEpsilon : theta - kEpsilon;
                auto pos1 = SphericalToCartesian(theta_1, phi);
                auto texcoord1 = Vector2(texcoord.x, theta_1 * kPiInv);

                auto phi2 = phi + kEpsilon < 2 * kPi ? phi + kEpsilon : phi - kEpsilon;
                auto pos2 = SphericalToCartesian(theta, phi2);
                auto texcoord2 = Vector2(phi2 * 0.5 * kPiInv, texcoord.y);

                auto v0v1 = pos1 - pos,
                     v0v2 = pos2 - pos;
                auto delta_uv_1 = texcoord1 - texcoord,
                     delta_uv_2 = texcoord2 - texcoord;

                auto r = 1 / (delta_uv_2.x * delta_uv_1.y - delta_uv_1.x * delta_uv_2.y);
                auto tangent = glm::normalize(Vector3(delta_uv_1.y * v0v2 - delta_uv_2.y * v0v1) * r),
                     bitangent = glm::normalize(Vector3(delta_uv_2.x * v0v1 - delta_uv_1.x * v0v2) * r);

                normal = this->material_->PerturbNormal(normal, tangent, bitangent, texcoord);
            }
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

        if (to_world_)
        {
            pos = TransfromPt(*to_world_, pos);
            normal = TransfromDir(*to_world_norm_, normal);
        }

        auto distance = glm::length(pos - ray.origin());
        return Intersection(pos, normal, texcoord, inside, distance, this->material_, area_);
    }

    std::pair<Intersection, Float> SampleP() const override
    {
        auto normal = SphereUniform();
        auto pos = center_ + radius_ * normal;

        if (to_world_)
        {
            pos = TransfromPt(*to_world_, pos);
            normal = TransfromDir(*to_world_norm_, normal);
        }

        return {Intersection(pos, normal, Vector2(-1), false, INFINITY, this->material_, area_), area_inv_};
    }

private:
    Vector3 center_;                 //球心
    Float radius_;                   //半径
    Material *material_;             //材质
    std::unique_ptr<Mat4> to_world_; //从局部坐标系到世界坐标系的变换矩阵

    std::unique_ptr<Mat4> to_world_norm_;
    std::unique_ptr<Mat4> to_local_;
    Float area_inv_;
};

NAMESPACE_END(simple_renderer)