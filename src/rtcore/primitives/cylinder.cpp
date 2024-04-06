#include "csrt/rtcore/primitives/cylinder.hpp"

#include "csrt/renderer/bsdfs/bsdf.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H AABB GetAabbCylinder(const CylinderData &data)
{
    AABB aabb;
    aabb += TransformPoint(data.to_world, Vec3{data.radius, data.radius, 0});
    aabb += TransformPoint(data.to_world, Vec3{-data.radius, -data.radius, 0});
    aabb += TransformPoint(data.to_world,
                           Vec3{data.radius, data.radius, data.length});
    aabb += TransformPoint(data.to_world,
                           Vec3{-data.radius, -data.radius, data.length});
    return aabb;
}

QUALIFIER_D_H bool IntersectCylinder(const uint32_t id_primitive,
                                     const CylinderData &data, Bsdf *bsdf,
                                     uint32_t *seed, Ray *ray, Hit *hit)
{
    const Mat4 to_local = data.to_world.Inverse();
    const Vec3 ray_origin = TransformPoint(to_local, ray->origin),
               ray_direction = TransformVector(to_local, ray->dir);
    const float a = Sqr(ray_direction.x) + Sqr(ray_direction.y),
                b = 2.0f * (ray_direction.x * ray_origin.x +
                            ray_direction.y * ray_origin.y),
                c = Sqr(ray_origin.x) + Sqr(ray_origin.y) - Sqr(data.radius);
    float t_near = 0.0f, t_far = 0.0f;
    if (!SolveQuadratic(a, b, c, &t_near, &t_far) || t_far < kEpsilonDistance)
        return false;

    const float z_near = ray_origin.z + ray_direction.z * t_near,
                z_far = ray_origin.z + ray_direction.z * t_far;
    float t = 0;
    if (kEpsilonDistance < t_near && 0.0f <= z_near && z_near <= data.length)
        t = t_near;
    else if (0.0 <= z_far && z_far <= data.length)
        t = t_far;
    else
        return false;

    const Vec3 position_local = ray_origin + t * ray_direction;
    const Vec2 texcoord = {atan2f(position_local.y, position_local.x) *
                               k1Div2Pi,
                           position_local.z / data.length};
    if (bsdf != nullptr && bsdf->IsTransparent(texcoord, seed))
        return false;

    const Vec3 position = TransformPoint(data.to_world, position_local);
    t = Length(position - ray->origin);
    if (t > ray->t_max || t < ray->t_min)
        return false;

    ray->t_max = t;

    if (hit != nullptr)
    {
        const bool inside = c < 0.0f;

        const Mat4 normal_to_world = data.to_world.Transpose().Inverse();
        const Vec3 normal_local =
            Normalize(Vec3{position_local.x, position_local.y, 0.0f});
        Vec3 normal = TransformVector(normal_to_world, normal_local),
             tangent = TransformVector(normal_to_world, {0, 0, 1}),
             bitangent = Normalize(Cross(normal, tangent));

        if (bsdf != nullptr)
        {
            normal =
                bsdf->ApplyBumpMapping(normal, tangent, bitangent, texcoord);
            bitangent = Normalize(Cross(normal, tangent));
            tangent = Normalize(Cross(bitangent, normal));
        }

        if (inside)
        {
            normal = -normal;
            bitangent = -bitangent;
        }

        *hit = Hit(id_primitive, inside, texcoord, position, normal, tangent,
                   bitangent);
    }

    return true;
}

QUALIFIER_D_H Hit SampleCylinder(const uint32_t id_primitive,
                                 const CylinderData &data, const float xi_0,
                                 const float xi_1)
{
    const float phi = k2Pi * xi_0, z = xi_1 * data.length;

    const Vec2 texcoord = {xi_0, xi_1};
    const Vec3 position = TransformPoint(
        data.to_world, {cosf(phi) * data.radius, sinf(phi) * data.radius, z});
    const Mat4 normal_to_world = data.to_world.Transpose().Inverse();
    const Vec3 normal =
        TransformVector(normal_to_world, {cosf(phi), sinf(phi), 0});
    return Hit(id_primitive, texcoord, position, normal);
}

} // namespace csrt