#include "primitive.cuh"

#include "../utils/math.cuh"
#include "../renderer/intersection.cuh"

QUALIFIER_DEVICE Primitive::Primitive()
    : type_(Type::kTriangle), id_bsdf_(kInvalidId), id_instance_(kInvalidId),
      pdf_area_(0), area_(0)
{
}

QUALIFIER_DEVICE Primitive::Primitive(Vertex *v, const uint64_t id_bsdf)
    : type_(Type::kTriangle), id_bsdf_(id_bsdf), id_instance_(kInvalidId),
      pdf_area_(0)
{
    for (int i = 0; i < 3; ++i)
    {
        geom_triangle_.texcoords[i] = v[i].texcoord;
        geom_triangle_.positions[i] = v[i].pos;
        geom_triangle_.normals[i] = v[i].normal;
        geom_triangle_.tangents[i] = v[i].tangent;
        geom_triangle_.bitangents[i] = v[i].bitangent;
    }
    geom_triangle_.v0v1 = v[1].pos - v[0].pos;
    geom_triangle_.v0v2 = v[2].pos - v[0].pos;

    area_ = Length(Cross(geom_triangle_.v0v1, geom_triangle_.v0v2)) * 0.5f;
}

QUALIFIER_DEVICE Primitive::Primitive(const Vec3 &center, const float radius, const Mat4 &to_world,
                                      const uint64_t id_bsdf)
    : type_(Type::kSphere), id_bsdf_(id_bsdf), id_instance_(kInvalidId),
      pdf_area_(0)
{
    geom_sphere_.center = center;
    geom_sphere_.radius = radius;
    geom_sphere_.to_world = to_world;
    geom_sphere_.to_local = to_world.Inverse();
    geom_sphere_.normal_to_world = to_world.Transpose().Inverse();

    const Vec3 center_world = TransfromPoint(to_world, center);
    const Vec3 p = TransfromPoint(to_world, center + Vec3(radius, 0.0f, 0.0f));
    const float radius_world = Length(center_world - p);
    area_ = (4.0f * kPi) * (radius_world * radius_world);
}

Primitive::Primitive(const Mat4 &to_world, const uint64_t id_bsdf)
    : type_(Type::kDisk), id_bsdf_(id_bsdf), id_instance_(kInvalidId),
      pdf_area_(0)
{
    geom_disk_.to_world = to_world;
    geom_disk_.normal_to_world = to_world.Transpose().Inverse();
    geom_disk_.to_local = to_world.Inverse();

    const Vec3 center = TransfromPoint(to_world, Vec3{0}),
               p1 = TransfromPoint(to_world, Vec3{0.5f, 0, 0}),
               p_min = TransfromPoint(to_world, Vec3{-0.5f, -0.5f, 0}),
               p_max = TransfromPoint(to_world, Vec3{0.5f, 0.5f, 0});
    const float radius = Length(p1 - center);
    area_ = kPi * radius * radius;
}

QUALIFIER_DEVICE void Primitive::SamplePoint(uint64_t *seed, Intersection *its) const
{
    switch (type_)
    {
    case Type::kTriangle:
    {
        const float a = sqrt(1.0f - RandomFloat(seed)),
                    beta = 1.0f - a,
                    gamma = a * RandomFloat(seed),
                    alpha = 1.0f - beta - gamma;

        const Vec2(&texcoords)[3] = geom_triangle_.texcoords;
        const Vec2 texcoord = alpha * texcoords[0] + beta * texcoords[1] + gamma * texcoords[2];

        const Vec3(&postions)[3] = geom_triangle_.positions;
        const Vec3 pos = alpha * postions[0] + beta * postions[1] + gamma * postions[2];

        const Vec3(&normals)[3] = geom_triangle_.normals;
        const Vec3 normal = alpha * normals[0] + beta * normals[1] + gamma * normals[2];

        *its = Intersection(id_instance_, false, texcoord, pos, normal, INFINITY, id_bsdf_,
                            pdf_area_);
        break;
    }
    case Type::kSphere:
    {
        const Vec3 vec = SampleSphereUniform(RandomFloat(seed), RandomFloat(seed)),
                   pos = TransfromPoint(geom_sphere_.to_world,
                                        geom_sphere_.center + geom_sphere_.radius * vec),
                   normal = TransfromVector(geom_sphere_.normal_to_world, vec);
        *its = Intersection(id_instance_, false, Vec2(0), pos, normal, INFINITY, id_bsdf_,
                            pdf_area_);
        break;
    }
    case Type::kDisk:
    {
        const Vec2 xy = SampleDiskUnifrom(RandomFloat(seed), RandomFloat(seed));
        const Vec3 pos = TransfromPoint(geom_disk_.to_world, {xy.u * 0.5f, xy.v * 0.5f, 0}),
                   normal = TransfromVector(geom_disk_.normal_to_world, {0, 0, 1});
        *its = Intersection(id_instance_, false, Vec2(0), pos, normal, INFINITY, id_bsdf_, pdf_area_);
        break;
    }
    }
}

QUALIFIER_DEVICE void Primitive::Intersect(const Ray &ray, Bsdf **bsdf_buffer, Texture **texture_buffer,
                                           const float *pixel_buffer, uint64_t *seed,
                                           Intersection *its) const
{
    switch (type_)
    {
    case Type::kTriangle:
        IntersectTriangle(ray, bsdf_buffer, texture_buffer, pixel_buffer, seed, its);
        break;
    case Type::kSphere:
        IntersectSphere(ray, bsdf_buffer, texture_buffer, pixel_buffer, seed, its);
        break;
    case Type::kDisk:
        IntersectDisk(ray, bsdf_buffer, texture_buffer, pixel_buffer, seed, its);
        break;
    }
}

QUALIFIER_DEVICE AABB Primitive::GetAabb() const
{
    AABB aabb;
    switch (type_)
    {
    case Type::kTriangle:
    {
        for (int i = 0; i < 3; ++i)
            aabb += geom_triangle_.positions[i];
        break;
    }
    case Type::kSphere:
    {
        aabb += TransfromPoint(geom_sphere_.to_world,
                               geom_sphere_.center + geom_sphere_.radius);
        aabb += TransfromPoint(geom_sphere_.to_world,
                               geom_sphere_.center - geom_sphere_.radius);
        break;
    }
    case Type::kDisk:
    {
        aabb += TransfromPoint(geom_disk_.to_world, Vec3{-0.5f, -0.5f, 0});
        aabb += TransfromPoint(geom_disk_.to_world, Vec3{0.5f, 0.5f, 0});
        break;
    }
    }
    return aabb;
}

QUALIFIER_DEVICE void Primitive::IntersectTriangle(const Ray &ray, Bsdf **bsdf_buffer,
                                                   Texture **texture_buffer, const float *pixel_buffer,
                                                   uint64_t *seed, Intersection *its) const
{
    const Vec3 P = Cross(ray.dir, geom_triangle_.v0v2);

    const float det = Dot(geom_triangle_.v0v1, P);
    if (fabs(det) < kEpsilonFloat)
        return;

    const float det_inv = 1.0f / det;
    const Vec3 T = ray.origin - geom_triangle_.positions[0],
               Q = Cross(T, geom_triangle_.v0v1);

    const float u = Dot(T, P) * det_inv;
    if (u < 0.0f || u > 1.0f)
        return;

    const float v = Dot(ray.dir, Q) * det_inv;
    if (v < 0.0f || (u + v) > 1.0f)
        return;

    const float distance = Dot(geom_triangle_.v0v2, Q) * det_inv;
    if (its->distance() < distance || distance < kEpsilonDistance)
        return;

    const bool inside = det < 0.0f;

    const float alpha = 1.0f - u - v,
                &beta = u,
                &gamma = v;
    const Vec2(&texcoords)[3] = geom_triangle_.texcoords;
    const Vec2 texcoord = alpha * texcoords[0] + beta * texcoords[1] + gamma * texcoords[2];

    Bsdf **bsdf = nullptr;
    if (id_bsdf_ != kInvalidId)
    {
        bsdf = bsdf_buffer + id_bsdf_;
        if (inside && !(*bsdf)->IsTwosided() && distance < its->distance())
        {
            *its = Intersection(id_instance_, distance);
            return;
        }
        if ((*bsdf)->IsTransparent(texcoord, pixel_buffer, texture_buffer, seed))
            return;
    }

    const Vec3(&postions)[3] = geom_triangle_.positions;
    const Vec3 pos = alpha * postions[0] + beta * postions[1] + gamma * postions[2];

    const Vec3(&normals)[3] = geom_triangle_.normals;
    Vec3 normal = Normalize(alpha * normals[0] + beta * normals[1] + gamma * normals[2]);
    if (bsdf != nullptr)
    {
        const Vec3(&tangents)[3] = geom_triangle_.tangents;
        Vec3 tangent = Normalize(alpha * tangents[0] + beta * tangents[1] + gamma * tangents[2]);

        const Vec3(&bitangents)[3] = geom_triangle_.bitangents;
        Vec3 bitangent = Normalize(alpha * bitangents[0] + beta * bitangents[1] + gamma * bitangents[2]);

        normal = (*bsdf)->ApplyBumpMapping(normal, tangent, bitangent, texcoord, pixel_buffer,
                                           texture_buffer, seed);
    }
    if (inside)
    {
        normal = -normal;
    }

    *its = Intersection(id_instance_, inside, texcoord, pos, normal, distance, id_bsdf_, pdf_area_);
}

QUALIFIER_DEVICE void Primitive::IntersectSphere(const Ray &ray, Bsdf **bsdf_buffer,
                                                 Texture **texture_buffer, const float *pixel_buffer,
                                                 uint64_t *seed, Intersection *its) const
{
    const Vec3 ray_origin_local = (TransfromPoint(geom_sphere_.to_local, ray.origin) -
                                   geom_sphere_.center),
               ray_direction_local = TransfromVector(geom_sphere_.to_local, ray.dir);

    const float a = Dot(ray_direction_local, ray_direction_local),
                b = 2.0f * Dot(ray_direction_local, ray_origin_local),
                c = (Dot(ray_origin_local, ray_origin_local) -
                     geom_sphere_.radius * geom_sphere_.radius);
    float t_near = 0.0f, t_far = 0.0f;
    if (!SolveQuadratic(a, b, c, t_near, t_far) || t_far < kEpsilonFloat)
        return;

    const float t = t_near < kEpsilonFloat ? t_far : t_near;

    const Vec3 pos_local = (geom_sphere_.center + ray_origin_local) + t * ray_direction_local,
               pos = TransfromPoint(geom_sphere_.to_world, pos_local);
    const float distance = Length(pos - ray.origin);
    if (its->distance() < distance)
        return;

    const bool inside = c < 0.0f;

    const Vec3 normal_local = Normalize(ray_origin_local + t * ray_direction_local);
    float theta, phi;
    CartesianToSpherical(normal_local, &theta, &phi, nullptr);

    const Vec2 texcoord = {phi * kOneDivTwoPi, theta * kPiInv};

    Bsdf **bsdf = nullptr;
    if (id_bsdf_ != kInvalidId)
    {
        bsdf = bsdf_buffer + id_bsdf_;
        if (inside && !(*bsdf)->IsTwosided())
        {
            *its = Intersection(id_instance_, distance);
            return;
        }
        if ((*bsdf)->IsTransparent(texcoord, pixel_buffer, texture_buffer, seed))
            return;
    }

    Vec3 normal = TransfromVector(geom_sphere_.normal_to_world, normal_local);

    if (bsdf != nullptr)
    {

        constexpr float epsilon_jitter = 0.01f * kPi;
        const float theta_prime = theta + epsilon_jitter < 0.5f * kPi
                                      ? theta + epsilon_jitter
                                      : theta - epsilon_jitter,
                    phi_prime = phi + epsilon_jitter < 2.0f * kPi
                                    ? phi + epsilon_jitter
                                    : phi - epsilon_jitter;
        const Vec3 p1 = TransfromPoint(geom_sphere_.to_world, SphericalToCartesian(theta_prime, phi, 1)),
                   p2 = TransfromPoint(geom_sphere_.to_world, SphericalToCartesian(theta, phi_prime, 1)),
                   v0v1 = p1 - pos,
                   v0v2 = p2 - pos;
        const Vec2 texcoord_1 = {texcoord.u, theta_prime * kPiInv},
                   texcoord_2 = {phi_prime * kOneDivTwoPi, texcoord.v},
                   delta_uv_1 = texcoord_1 - texcoord,
                   delta_uv_2 = texcoord_2 - texcoord;
        const float r = 1.0f / (delta_uv_2.u * delta_uv_1.v - delta_uv_1.u * delta_uv_2.v);
        const Vec3 tangent = Normalize(r * Vec3{delta_uv_1.v * v0v2 - delta_uv_2.v * v0v1}),
                   bitangent = Normalize(Cross(tangent, normal));
        normal = (*bsdf)->ApplyBumpMapping(normal, tangent, bitangent, texcoord, pixel_buffer,
                                           texture_buffer, seed);
    }
    if (inside)
    {
        normal = -normal;
    }

    *its = Intersection(id_instance_, inside, texcoord, pos, normal, distance, id_bsdf_, pdf_area_);
}

QUALIFIER_DEVICE void Primitive::IntersectDisk(const Ray &ray, Bsdf **bsdf_buffer,
                                               Texture **texture_buffer, const float *pixel_buffer,
                                               uint64_t *seed, Intersection *its) const
{
    const Vec3 ray_origin_local = TransfromPoint(geom_disk_.to_local, ray.origin),
               ray_direction_local = TransfromVector(geom_disk_.to_local, ray.dir);

    float t_z = -ray_origin_local.z / ray_direction_local.z;
    if (t_z < kEpsilonFloat)
        return;

    Vec3 pos_local = ray_origin_local + t_z * ray_direction_local;
    if (Length(pos_local) > 0.5f)
        return;
    Vec3 pos = TransfromPoint(geom_disk_.to_world, pos_local);
    const float distance = Length(pos - ray.origin);

    float theta, phi, r;
    CartesianToSpherical(pos_local, &theta, &phi, &r);
    const Vec2 texcoord = {r, phi * 0.5f * kPiInv};

    const bool inside = ray_direction_local.z > 0.0f;

    Bsdf **bsdf = nullptr;
    if (id_bsdf_ != kInvalidId)
    {
        bsdf = bsdf_buffer + id_bsdf_;
        if (inside && !(*bsdf)->IsTwosided())
        {
            *its = Intersection(id_instance_, distance);
            return;
        }
        if ((*bsdf)->IsTransparent(texcoord, pixel_buffer, texture_buffer, seed))
            return;
    }

    Vec3 normal = {0, 0, 1};
    if (bsdf != nullptr)
    {
        const float r_1 = r + kEpsilon < 1 ? r + kEpsilon : r - kEpsilon,
              phi2 = phi + kEpsilon < 2.0f * kPi ? phi + kEpsilon : phi - kEpsilon;
        const Vec3 pos1 = SphericalToCartesian(theta, phi, r_1),
             pos2 = SphericalToCartesian(theta, phi2, r);
        const Vec2 texcoord1 = {r_1, texcoord.v},
             texcoord2 = {texcoord.u, phi2 * kOneDivTwoPi};

        const Vec3 v0v1 = pos1 - pos_local,
             v0v2 = pos2 - pos_local;
        const Vec2 delta_uv_1 = texcoord1 - texcoord,
             delta_uv_2 = texcoord2 - texcoord;

        const float norm = 1.0f / (delta_uv_2.u * delta_uv_1.v - delta_uv_1.u * delta_uv_2.v);
        Vec3 tangent = Normalize((delta_uv_1.v * v0v2 - delta_uv_2.v * v0v1) * norm),
             bitangent = Normalize((delta_uv_2.u * v0v1 - delta_uv_1.u * v0v2) * norm);

        normal = (*bsdf)->ApplyBumpMapping(normal, tangent, bitangent, texcoord, pixel_buffer,
                                           texture_buffer, seed);
    }
    normal = TransfromVector(geom_disk_.normal_to_world, normal);
    if (inside)
    {
        normal = -normal;
    }

    *its = Intersection(id_instance_, inside, texcoord, pos, normal, distance, id_bsdf_, pdf_area_);
}
