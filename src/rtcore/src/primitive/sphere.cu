#include "primitive.cuh"

#include "utils.cuh"

namespace csrt
{

QUALIFIER_D_H AABB Primitive::GetAabbSphere() const
{
    AABB aabb;
    aabb += TransformPoint(data_.sphere.to_world,
                           data_.sphere.center + data_.sphere.radius);
    aabb += TransformPoint(data_.sphere.to_world,
                           data_.sphere.center - data_.sphere.radius);
    return aabb;
}

QUALIFIER_D_H void Primitive::IntersectSphere(Ray *ray, Hit *hit) const
{
    const Vec3 ray_origin = TransformPoint(data_.sphere.to_local, ray->origin) -
                            data_.sphere.center,
               ray_direction = TransformVector(data_.sphere.to_local, ray->dir);
    const float a = Dot(ray_direction, ray_direction),
                b = 2.0f * Dot(ray_direction, ray_origin),
                c = Dot(ray_origin, ray_origin) - Sqr(data_.sphere.radius);
    float t_near = 0.0f, t_far = 0.0f;
    if (!SolveQuadratic(a, b, c, &t_near, &t_far) || t_far < kEpsilonDistance)
        return;

    float t = t_near < kEpsilonDistance ? t_far : t_near;
    const Vec3 position_local = ray_origin + t * ray_direction,
               position = TransformPoint(data_.sphere.to_world,
                                         position_local + data_.sphere.center);
    t = Length(position - ray->origin);
    if (t > ray->t_max || t < ray->t_min)
        return;
    ray->t_max = t;

    float theta, phi;
    CartesianToSpherical(position_local, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};

    const bool inside = c < 0.0f;

    const Vec3 normal_local = Normalize(position_local);
    Vec3 normal = TransformVector(data_.sphere.normal_to_world, normal_local);

    constexpr float epsilon_jitter = 0.01f * kPi;
    float theta_prime = theta + epsilon_jitter;
    const bool flip_bitangent = theta_prime > kPi;
    if (flip_bitangent)
        theta_prime = theta - epsilon_jitter;
    const Vec3 position_prime = TransformPoint(
        data_.sphere.to_world, SphericalToCartesian(theta_prime, phi, 1));
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

    *hit = Hit(id_, inside, texcoord, position, normal, tangent, bitangent);
}

QUALIFIER_D_H Hit Primitive::SampleSphere(const float xi_0,
                                          const float xi_1) const
{
    const float cos_theta = 1.0f - 2.0f * xi_0;
    const Vec2 texcoord = {xi_1, acosf(cos_theta) * k1DivPi};
    const float sin_theta = sqrtf(1.0f - Sqr(cos_theta)), phi = k2Pi * xi_1;
    const Vec3 normal_local = {sin_theta * cosf(phi), sin_theta * sinf(phi),
                               cos_theta},
               position_local =
                   data_.sphere.center + data_.sphere.radius * normal_local;
    const Vec3 position = TransformPoint(data_.sphere.to_world, position_local),
               normal =
                   TransformVector(data_.sphere.normal_to_world, normal_local);
    return Hit(id_, texcoord, position, normal);
}

} // namespace csrt