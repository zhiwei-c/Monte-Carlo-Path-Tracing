#include "csrt/rtcore/primitives/sphere.cuh"

#include "csrt/renderer/bsdfs/bsdf.cuh"
#include "csrt/utils.cuh"

namespace csrt
{

QUALIFIER_D_H AABB GetAabbSphere(const SphereData &data)
{
    AABB aabb;
    aabb += TransformPoint(data.to_world, data.center + data.radius);
    aabb += TransformPoint(data.to_world, data.center - data.radius);
    return aabb;
}

QUALIFIER_D_H bool IntersectSphere(const uint32_t id_primitive,
                                   const SphereData &data, Bsdf *bsdf,
                                   uint32_t *seed, Ray *ray, Hit *hit)
{
    const Mat4 to_local = data.to_world.Inverse();
    const Vec3 ray_origin = TransformPoint(to_local, ray->origin) - data.center,
               ray_direction = TransformVector(to_local, ray->dir);
    const float a = Dot(ray_direction, ray_direction),
                b = 2.0f * Dot(ray_direction, ray_origin),
                c = Dot(ray_origin, ray_origin) - Sqr(data.radius);
    float t_near = 0.0f, t_far = 0.0f;
    if (!SolveQuadratic(a, b, c, &t_near, &t_far) || t_far < kEpsilonDistance)
        return false;

    float t = t_near < kEpsilonDistance ? t_far : t_near;
    const Vec3 position_local = ray_origin + t * ray_direction,
               position =
                   TransformPoint(data.to_world, position_local + data.center);
    t = Length(position - ray->origin);
    if (t > ray->t_max || t < ray->t_min)
        return false;

    float theta, phi;
    CartesianToSpherical(position_local, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    if (bsdf->IsTransparent(texcoord, seed))
        return false;

    ray->t_max = t;
    
    if (hit != nullptr)
    {
        const bool inside = c < 0.0f;

        const Mat4 normal_to_world = data.to_world.Transpose().Inverse();
        const Vec3 normal_local = Normalize(position_local);
        Vec3 normal = TransformVector(normal_to_world, normal_local);

        constexpr float epsilon_jitter = 0.01f * kPi;
        float theta_prime = theta + epsilon_jitter;
        const bool flip_bitangent = theta_prime > kPi;
        if (flip_bitangent)
            theta_prime = theta - epsilon_jitter;
        const Vec3 position_prime = TransformPoint(
            data.to_world, SphericalToCartesian(theta_prime, phi, 1));
        Vec3 bitangent = Normalize(position_prime - position);
        if (flip_bitangent)
            bitangent = -bitangent;

        const Vec3 tangent = Normalize(Cross(bitangent, normal));
        bitangent = Normalize(Cross(normal, tangent));

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

QUALIFIER_D_H Hit SampleSphere(const uint32_t id_primitive,
                               const SphereData &data, const float xi_0,
                               const float xi_1)
{
    const float cos_theta = 1.0f - 2.0f * xi_0;
    const Vec2 texcoord = {xi_1, acosf(cos_theta) * k1DivPi};
    const float sin_theta = sqrtf(1.0f - Sqr(cos_theta)), phi = k2Pi * xi_1;
    const Vec3 normal_local = {sin_theta * cosf(phi), sin_theta * sinf(phi),
                               cos_theta},
               position_local = data.center + data.radius * normal_local;
    const Vec3 position = TransformPoint(data.to_world, position_local);

    const Mat4 normal_to_world = data.to_world.Transpose().Inverse();
    const Vec3 normal = TransformVector(normal_to_world, normal_local);

    return Hit(id_primitive, texcoord, position, normal);
}

} // namespace csrt