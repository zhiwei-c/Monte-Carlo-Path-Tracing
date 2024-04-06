#include "csrt/rtcore/primitives/disk.hpp"

#include "csrt/renderer/bsdfs/bsdf.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H AABB GetAabbDisk(const DiskData &data)
{
    AABB aabb;
    aabb += TransformPoint(data.to_world, Vec3{-0.5f, -0.5f, 0});
    aabb += TransformPoint(data.to_world, Vec3{0.5f, 0.5f, 0});
    return aabb;
}

QUALIFIER_D_H bool IntersectDisk(const uint32_t id_primitive,
                                 const DiskData &data, Bsdf *bsdf,
                                 uint32_t *seed, Ray *ray, Hit *hit)
{
    const Mat4 to_local = data.to_world.Inverse();
    const Vec3 ray_origin = TransformPoint(to_local, ray->origin),
               ray_direction = TransformVector(to_local, ray->dir);

    const float t_z = -ray_origin.z / ray_direction.z;
    if (t_z < kEpsilonFloat)
        return false;

    const Vec3 position_local = ray_origin + t_z * ray_direction;
    if (Length(position_local) > 0.5f)
        return false;

    const Vec3 position = TransformPoint(data.to_world, position_local);
    const float t = Length(position - ray->origin);
    if (t > ray->t_max || t < ray->t_min)
        return false;

    float theta, phi, r;
    CartesianToSpherical(position_local, &theta, &phi, &r);
    const Vec2 texcoord = {r, phi * k1Div2Pi};
    if (bsdf != nullptr && bsdf->IsTransparent(texcoord, seed))
        return false;

    ray->t_max = t;

    if (hit != nullptr)
    {
        const bool inside = ray_direction.z > 0;

        constexpr float epsilon_jitter = 0.01f * kPi;

        float r_prime = r + epsilon_jitter;
        const bool flip_bitangent = r_prime > r;
        if (flip_bitangent)
            r_prime = r - epsilon_jitter;

        float phi_prime = phi + epsilon_jitter;
        const bool flip_tangent = phi_prime > kPi;
        if (flip_tangent)
            phi_prime = phi - epsilon_jitter;

        const Vec3 v0v1_local = SphericalToCartesian(theta, phi, r_prime) -
                                position_local,
                   v0v2_local = SphericalToCartesian(theta, phi_prime, r) -
                                position_local;
        const Vec2 delta_uv_1 = Vec2{r_prime, texcoord.v} - texcoord,
                   delta_uv_2 =
                       Vec2{texcoord.u, phi_prime * k1Div2Pi} - texcoord;
        const float norm =
            1.0f / (delta_uv_2.u * delta_uv_1.v - delta_uv_1.u * delta_uv_2.v);
        Vec3 tangent = Normalize(
                 (delta_uv_1.v * v0v2_local - delta_uv_2.v * v0v1_local) *
                 norm),
             bitangent = Normalize(
                 (delta_uv_2.u * v0v1_local - delta_uv_1.u * v0v2_local) *
                 norm),
             normal = {0, 0, 1};
        if (flip_bitangent)
            bitangent = -bitangent;
        if (flip_tangent)
            tangent = -tangent;

        bitangent = Normalize(Cross(normal, tangent));
        tangent = Normalize(Cross(bitangent, normal));

        if (bsdf != nullptr)
        {
            normal =
                bsdf->ApplyBumpMapping(normal, tangent, bitangent, texcoord);
            bitangent = Normalize(Cross(normal, tangent));
            tangent = Normalize(Cross(bitangent, normal));
        }

        const Mat4 normal_to_world = data.to_world.Transpose().Inverse();
        normal = TransformVector(normal_to_world, normal);
        tangent = TransformVector(data.to_world, tangent);
        bitangent = TransformVector(data.to_world, bitangent);

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

QUALIFIER_D_H Hit SampleDisk(const uint32_t id_primitive, const DiskData &data,
                             const float xi_0, const float xi_1)
{
    const float r1 = 2.0f * xi_0 - 1.0f, r2 = 2.0f * xi_1 - 1.0f;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
     */
    float phi, r;
    if (r1 == 0.0f && r2 == 0.0f)
    {
        r = phi = 0;
    }
    else if (Sqr(r1) > Sqr(r2))
    {
        r = r1;
        phi = kPiDiv4 * (r2 / r1);
    }
    else
    {
        r = r2;
        phi = kPiDiv2 - (r1 / r2) * kPiDiv4;
    }
    const Vec2 xy =  {r * cosf(phi), r * sinf(phi)};

    const Vec2 texcoord = {r, phi * k1Div2Pi};
    const Vec3 position = TransformPoint(data.to_world, {xy.u * 0.5f, xy.v * 0.5f, 0});
    const Mat4 normal_to_world = data.to_world.Transpose().Inverse();
    const Vec3 normal = TransformVector(normal_to_world, {0, 0, 1});
    return Hit(id_primitive, texcoord, position, normal);
}

} // namespace csrt