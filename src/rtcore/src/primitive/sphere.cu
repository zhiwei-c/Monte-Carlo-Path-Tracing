#include "primitive.cuh"

#include "utils.cuh"

namespace rt
{

QUALIFIER_D_H AABB Primitive::GetAabbSphere() const
{
    AABB aabb;
    aabb +=
        Mul(geom_.sphere.to_world, {geom_.sphere.center + geom_.sphere.radius, 1.0f}).position();
    aabb +=
        Mul(geom_.sphere.to_world, {geom_.sphere.center - geom_.sphere.radius, 1.0f}).position();
    return aabb;
}

QUALIFIER_D_H void Primitive::IntersectSphere(Ray *ray, Hit *hit) const
{
    const Vec3 ray_origin =
                   Mul(geom_.sphere.to_local, {ray->origin, 1.0f}).position() - geom_.sphere.center,
               ray_direction = Mul(geom_.sphere.to_local, {ray->dir, 0.0f}).direction();
    const float a = Dot(ray_direction, ray_direction), b = 2.0f * Dot(ray_direction, ray_origin),
                c = Dot(ray_origin, ray_origin) - Sqr(geom_.sphere.radius);
    float t_near = 0.0f, t_far = 0.0f;
    if (!SolveQuadratic(a, b, c, &t_near, &t_far))
        return;

    float t = t_near < kEpsilonFloat ? t_far : t_near;
    const Vec3 position_local = ray_origin + t * ray_direction,
               position = Mul(geom_.sphere.to_world, {position_local + geom_.sphere.center, 1.0f})
                              .position();
    t = Length(position - ray->origin);
    if (t > ray->t_max || t < ray->t_min)
        return;
    ray->t_max = t;

    float theta, phi;
    CartesianToSpherical(position_local, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};

    const Vec3 normal_local = Normalize(position_local),
               normal = Mul(geom_.sphere.normal_to_world, {normal_local, 0.0f}).direction();

    constexpr float epsilon_jitter = 0.01f * kPi;
    float theta_prime = theta + epsilon_jitter;
    const bool flip_bitangent = theta_prime > kPi;
    if (flip_bitangent)
        theta_prime = theta - epsilon_jitter;
    Vec3 bitangent = Normalize(
        Mul(geom_.sphere.to_world, {SphericalToCartesian(theta_prime, phi, 1), 1.0f}).position() -
        position);
    if (flip_bitangent)
        bitangent = -bitangent;

    const Vec3 tangent = Normalize(Cross(bitangent, normal));

    const bool inside = c < 0.0f;

    *hit = Hit(id_primitive_, inside, texcoord, position, normal, tangent, bitangent);
}

QUALIFIER_D_H Hit Primitive::SampleSphere(const float xi_0, const float xi_1,
                                          const float xi_2) const
{
    const float cos_theta = 1.0f - 2.0f * xi_1;
    const Vec2 texcoord = {xi_2, acosf(cos_theta) * k1DivPi};
    const float sin_theta = sqrtf(1.0f - Sqr(cos_theta)), phi = k2Pi * xi_2;
    const Vec3 normal_local = {sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta},
               position_local = geom_.sphere.center + geom_.sphere.radius * normal_local;
    const Vec3 position = Mul(geom_.sphere.to_world, {position_local, 1.0f}).position(),
               normal = Mul(geom_.sphere.normal_to_world, {normal_local, 0.0f}).direction();
    return Hit(id_primitive_, texcoord, position, normal);
}

} // namespace rt