#include "primitive.cuh"

#include "../utils/math.cuh"

Primitive::Info Primitive::Info::CreateTriangle(Vertex *v, const uint32_t id_bsdf)
{
    Primitive::Info info;
    info.type = Type::kTriangle;
    info.data.id_bsdf = id_bsdf;
    for (int i = 0; i < 3; ++i)
    {
        info.data.triangle.texcoords[i] = v[i].texcoord;
        info.data.triangle.positions[i] = v[i].position;
        info.data.triangle.normals[i] = v[i].normal;
        info.data.triangle.tangents[i] = v[i].tangent;
        info.data.triangle.bitangents[i] = v[i].bitangent;
    }
    return info;
}

Primitive::Info Primitive::Info::CreateSphere(const Vec3 &center, const float radius,
                                              const Mat4 &to_world, const uint32_t id_bsdf)
{
    Primitive::Info info;
    info.type = Type::kSphere;
    info.data.id_bsdf = id_bsdf;
    info.data.sphere.center = center;
    info.data.sphere.radius = radius;
    info.data.sphere.to_world = to_world;
    return info;
}

Primitive::Info Primitive::Info::CreateDisk(const Mat4 &to_world, const uint32_t id_bsdf)
{
    Primitive::Info info;
    info.type = Type::kDisk;
    info.data.id_bsdf = id_bsdf;
    info.data.disk.to_world = to_world;
    return info;
}

Primitive::Primitive()
    : id_primitive_(kInvalidId), type_(Type::kInvalid), id_bsdf_(kInvalidId),
      id_instance_(kInvalidId)
{
}

Primitive::Primitive(const uint32_t id_primitive, const Info &info)
    : id_primitive_(id_primitive), type_(info.type), id_bsdf_(info.data.id_bsdf),
      id_instance_(kInvalidId)
{
    switch (type_)
    {
    case Type::kTriangle:
        InitTriangle(info.data.triangle);
        break;
    case Type::kSphere:
        InitSphere(info.data.sphere);
        break;
    case Type::kDisk:
        InitDisk(info.data.disk);
        break;
    }
}

QUALIFIER_DEVICE void Primitive::Intersect(const Ray &ray, Bsdf **bsdf_buffer,
                                           Texture **texture_buffer, const float *pixel_buffer,
                                           uint32_t *seed, Intersection *its) const
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

QUALIFIER_DEVICE void Primitive::SamplePoint(uint32_t *seed, Intersection *its) const
{
    switch (type_)
    {
    case Type::kTriangle:
    {
        const float a = sqrt(1.0f - RandomFloat(seed));
        const float beta = 1.0f - a, gamma = a * RandomFloat(seed), alpha = 1.0f - beta - gamma;
        const Vec2 texcoord = Lerp(geom_.triangle.texcoords, alpha, beta, gamma);
        const Vec3 position = Lerp(geom_.triangle.positions, alpha, beta, gamma),
                   normal = Lerp(geom_.triangle.normals, alpha, beta, gamma);
        *its = Intersection(id_instance_, texcoord, position, normal, id_bsdf_, pdf_area_);
        break;
    }
    case Type::kSphere:
    {
        const Vec3 vec = SampleSphereUniform(RandomFloat(seed), RandomFloat(seed)),
                   position = TransfromPoint(geom_.sphere.to_world,
                                             geom_.sphere.center + geom_.sphere.radius * vec),
                   normal = TransfromVector(geom_.sphere.normal_to_world, vec);
        *its = Intersection(id_instance_, Vec2(0), position, normal, id_bsdf_, pdf_area_);
        break;
    }
    case Type::kDisk:
    {
        const Vec2 xy = SampleDiskUnifrom(RandomFloat(seed), RandomFloat(seed));
        const Vec3 position = TransfromPoint(geom_.disk.to_world, {xy.u * 0.5f, xy.v * 0.5f, 0}),
                   normal = TransfromVector(geom_.disk.normal_to_world, {0, 0, 1});
        *its = Intersection(id_instance_, Vec2(0), position, normal, id_bsdf_, pdf_area_);
        break;
    }
    }
}

QUALIFIER_DEVICE AABB Primitive::aabb() const
{
    AABB aabb;
    switch (type_)
    {
    case Type::kTriangle:
    {
        for (int i = 0; i < 3; ++i)
            aabb += geom_.triangle.positions[i];
        break;
    }
    case Type::kSphere:
    {
        aabb += TransfromPoint(geom_.sphere.to_world,
                               geom_.sphere.center + geom_.sphere.radius);
        aabb += TransfromPoint(geom_.sphere.to_world,
                               geom_.sphere.center - geom_.sphere.radius);
        break;
    }
    case Type::kDisk:
    {
        aabb += TransfromPoint(geom_.disk.to_world, Vec3{-0.5f, -0.5f, 0});
        aabb += TransfromPoint(geom_.disk.to_world, Vec3{0.5f, 0.5f, 0});
        break;
    }
    }
    return aabb;
}

QUALIFIER_DEVICE void Primitive::InitDisk(const Info::Data::Disk &data)
{
    geom_.disk.to_world = data.to_world;
    geom_.disk.normal_to_world = data.to_world.Transpose().Inverse();
    geom_.disk.to_local = data.to_world.Inverse();

    const Vec3 center = TransfromPoint(data.to_world, Vec3{0}),
               p1 = TransfromPoint(data.to_world, Vec3{0.5f, 0, 0}),
               p_min = TransfromPoint(data.to_world, Vec3{-0.5f, -0.5f, 0}),
               p_max = TransfromPoint(data.to_world, Vec3{0.5f, 0.5f, 0});
    const float radius = Length(p1 - center);
    area_ = kPi * radius * radius;
}

QUALIFIER_DEVICE void Primitive::InitSphere(const Info::Data::Sphere &data)
{
    geom_.sphere.center = data.center;
    geom_.sphere.radius = data.radius;
    geom_.sphere.to_world = data.to_world;
    geom_.sphere.to_local = data.to_world.Inverse();
    geom_.sphere.normal_to_world = data.to_world.Transpose().Inverse();

    const Vec3 center_world = TransfromPoint(data.to_world, data.center);
    const Vec3 p = TransfromPoint(data.to_world, data.center + Vec3(data.radius, 0.0f, 0.0f));
    const float radius_world = Length(center_world - p);
    area_ = (4.0f * kPi) * (radius_world * radius_world);
}

QUALIFIER_DEVICE void Primitive::InitTriangle(const Info::Data::Triangle &data)
{
    for (int i = 0; i < 3; ++i)
    {
        geom_.triangle.texcoords[i] = data.texcoords[i];
        geom_.triangle.positions[i] = data.positions[i];
        geom_.triangle.normals[i] = data.normals[i];
        geom_.triangle.tangents[i] = data.tangents[i];
        geom_.triangle.bitangents[i] = data.bitangents[i];
    }
    geom_.triangle.v0v1 = data.positions[1] - data.positions[0];
    geom_.triangle.v0v2 = data.positions[2] - data.positions[0];

    area_ = Length(Cross(geom_.triangle.v0v1, geom_.triangle.v0v2)) * 0.5f;
}

QUALIFIER_DEVICE void Primitive::IntersectDisk(const Ray &ray, Bsdf **bsdf_buffer,
                                               Texture **texture_buffer, const float *pixel_buffer,
                                               uint32_t *seed, Intersection *its) const
{
    const Vec3 ray_origin_local = TransfromPoint(geom_.disk.to_local, ray.origin),
               ray_direction_local = TransfromVector(geom_.disk.to_local, ray.dir);

    float t_z = -ray_origin_local.z / ray_direction_local.z;
    if (t_z < kEpsilonFloat)
        return;

    Vec3 position_local = ray_origin_local + t_z * ray_direction_local;
    if (Length(position_local) > 0.5f)
        return;

    Vec3 position = TransfromPoint(geom_.disk.to_world, position_local);
    const float distance = Length(position - ray.origin);
    if (its->distance < distance)
        return;

    float theta, phi, r;
    CartesianToSpherical(position_local, &theta, &phi, &r);
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
        if ((*bsdf)->IsTransparent(texcoord, texture_buffer, pixel_buffer, seed))
            return;
    }

    const float r_1 = r + kEpsilon < 1 ? r + kEpsilon : r - kEpsilon,
                phi2 = phi + kEpsilon < 2.0f * kPi ? phi + kEpsilon : phi - kEpsilon;
    const Vec3 v0v1_local = SphericalToCartesian(theta, phi, r_1) - position_local,
               v0v2_local = SphericalToCartesian(theta, phi2, r) - position_local;
    const Vec2 delta_uv_1 = Vec2{r_1, texcoord.v} - texcoord,
               delta_uv_2 = Vec2{texcoord.u, phi2 * kOneDivTwoPi} - texcoord;
    const float norm = 1.0f / (delta_uv_2.u * delta_uv_1.v - delta_uv_1.u * delta_uv_2.v);
    Vec3 tangent = Normalize((delta_uv_1.v * v0v2_local - delta_uv_2.v * v0v1_local) * norm),
         bitangent = Normalize((delta_uv_2.u * v0v1_local - delta_uv_1.u * v0v2_local) * norm),
         normal = {0, 0, 1};
    if (bsdf != nullptr)
    {
        normal = (*bsdf)->ApplyBumpMapping(normal, tangent, bitangent, texcoord, texture_buffer,
                                           pixel_buffer, seed);
    }
    normal = TransfromVector(geom_.disk.normal_to_world, normal);
    tangent = TransfromVector(geom_.disk.to_world, tangent);
    bitangent = TransfromVector(geom_.disk.to_world, bitangent);
    if (inside)
    {
        normal = -normal;
        tangent = -tangent;
        bitangent = -bitangent;
    }

    *its = Intersection(id_instance_, inside, texcoord, position, normal, tangent, bitangent,
                        distance, id_bsdf_, pdf_area_);
}

QUALIFIER_DEVICE void Primitive::IntersectSphere(const Ray &ray, Bsdf **bsdf_buffer,
                                                 Texture **texture_buffer,
                                                 const float *pixel_buffer,
                                                 uint32_t *seed, Intersection *its) const
{
    const Vec3 ray_origin_local = (TransfromPoint(geom_.sphere.to_local, ray.origin) -
                                   geom_.sphere.center),
               ray_direction_local = TransfromVector(geom_.sphere.to_local, ray.dir);

    const float a = Dot(ray_direction_local, ray_direction_local),
                b = 2.0f * Dot(ray_direction_local, ray_origin_local),
                c = (Dot(ray_origin_local, ray_origin_local) -
                     geom_.sphere.radius * geom_.sphere.radius);
    float t_near = 0.0f, t_far = 0.0f;
    if (!SolveQuadratic(a, b, c, t_near, t_far) || t_far < kEpsilonFloat)
        return;
    const float t = t_near < kEpsilonFloat ? t_far : t_near;

    const Vec3 position_local = (geom_.sphere.center + ray_origin_local) + t * ray_direction_local,
               position = TransfromPoint(geom_.sphere.to_world, position_local);
    const float distance = Length(position - ray.origin);
    if (its->distance < distance)
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
        if ((*bsdf)->IsTransparent(texcoord, texture_buffer, pixel_buffer, seed))
            return;
    }

    constexpr float epsilon_jitter = 0.01f * kPi;
    const float theta_prime = theta + epsilon_jitter < 0.5f * kPi ? theta + epsilon_jitter
                                                                  : theta - epsilon_jitter,
                phi_prime = phi + epsilon_jitter < 2.0f * kPi ? phi + epsilon_jitter
                                                              : phi - epsilon_jitter;
    const Vec3 v0v1 = (TransfromPoint(geom_.sphere.to_world,
                                      SphericalToCartesian(theta_prime, phi, 1)) -
                       position),
               v0v2 = (TransfromPoint(geom_.sphere.to_world,
                                      SphericalToCartesian(theta, phi_prime, 1)) -
                       position);
    const Vec2 delta_uv_1 = Vec2{texcoord.u, theta_prime * kPiInv} - texcoord,
               delta_uv_2 = Vec2{phi_prime * kOneDivTwoPi, texcoord.v} - texcoord;
    const float r = 1.0f / (delta_uv_2.u * delta_uv_1.v - delta_uv_1.u * delta_uv_2.v);
    Vec3 tangent = Normalize(r * Vec3{delta_uv_1.v * v0v2 - delta_uv_2.v * v0v1}),
         normal = TransfromVector(geom_.sphere.normal_to_world, normal_local),
         bitangent = Normalize(Cross(tangent, normal));
    if (bsdf != nullptr)
    {
        normal = (*bsdf)->ApplyBumpMapping(normal, tangent, bitangent, texcoord, texture_buffer,
                                           pixel_buffer, seed);
    }
    if (inside)
    {
        normal = -normal;
        tangent = -tangent;
        bitangent = -bitangent;
    }

    *its = Intersection(id_instance_, inside, texcoord, position, normal, tangent, bitangent,
                        distance, id_bsdf_, pdf_area_);
}

QUALIFIER_DEVICE void Primitive::IntersectTriangle(const Ray &ray, Bsdf **bsdf_buffer,
                                                   Texture **texture_buffer,
                                                   const float *pixel_buffer, uint32_t *seed,
                                                   Intersection *its) const
{
    const Vec3 P = Cross(ray.dir, geom_.triangle.v0v2);

    const float det = Dot(geom_.triangle.v0v1, P);
    if (fabs(det) < kEpsilonFloat)
        return;

    const float det_inv = 1.0f / det;
    const Vec3 T = ray.origin - geom_.triangle.positions[0],
               Q = Cross(T, geom_.triangle.v0v1);

    const float u = Dot(T, P) * det_inv;
    if (u < 0.0f || u > 1.0f)
        return;

    const float v = Dot(ray.dir, Q) * det_inv;
    if (v < 0.0f || (u + v) > 1.0f)
        return;

    const float distance = Dot(geom_.triangle.v0v2, Q) * det_inv;
    if (its->distance < distance || distance < kEpsilonDistance)
        return;

    const float alpha = 1.0f - u - v, &beta = u, &gamma = v;
    const Vec2 texcoord = Lerp(geom_.triangle.texcoords, alpha, beta, gamma);

    const bool inside = det < 0.0f;
    Bsdf **bsdf = nullptr;
    if (id_bsdf_ != kInvalidId)
    {
        bsdf = bsdf_buffer + id_bsdf_;
        if (inside && !(*bsdf)->IsTwosided())
        {
            *its = Intersection(id_instance_, distance);
            return;
        }
        if ((*bsdf)->IsTransparent(texcoord, texture_buffer, pixel_buffer, seed))
            return;
    }

    const Vec3 position = Lerp(geom_.triangle.positions, alpha, beta, gamma);
    Vec3 normal = Normalize(Lerp(geom_.triangle.normals, alpha, beta, gamma)),
         tangent = Normalize(Lerp(geom_.triangle.tangents, alpha, beta, gamma)),
         bitangent = Normalize(Lerp(geom_.triangle.bitangents, alpha, beta, gamma));
    bitangent = Normalize(Cross(normal, tangent));
    tangent = Normalize(Cross(bitangent, normal));

    if (bsdf != nullptr)
    {
        normal = (*bsdf)->ApplyBumpMapping(normal, tangent, bitangent, texcoord, texture_buffer,
                                           pixel_buffer, seed);
    }
    if (inside)
    {
        normal = -normal;
        tangent = -tangent;
        bitangent = -bitangent;
    }

    *its = Intersection(id_instance_, inside, texcoord, position, normal, tangent, bitangent,
                        distance, id_bsdf_, pdf_area_);
}
