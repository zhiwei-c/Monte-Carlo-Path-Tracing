#pragma once

#include "../global.cuh"
#include "../accelerators/aabb.cuh"
#include "../renderer/ray.cuh"
#include "../renderer/intersection.cuh"

struct Vertex
{
    Vec2 texcoord;
    Vec3 pos;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
};

class Primitive
{
public:
    enum Type
    {
        kTriangle,
        kSphere,
        kDisk,
    };

    QUALIFIER_DEVICE Primitive();
    QUALIFIER_DEVICE Primitive(Vertex *v, const uint64_t id_bsdf);
    QUALIFIER_DEVICE Primitive(const Vec3 &center, const float radius, const Mat4 &to_world,
                               const uint64_t id_bsdf);
    QUALIFIER_DEVICE Primitive(const Mat4 &to_world, const uint64_t id_bsdf);

    QUALIFIER_DEVICE void SamplePoint(uint64_t *seed, Intersection *its) const;
    QUALIFIER_DEVICE void Intersect(const Ray &ray, Bsdf **bsdf_buffer, Texture **texture_buffer,
                                    const float *pixel_buffer, uint64_t *seed, Intersection *its) const;

    QUALIFIER_DEVICE AABB GetAabb() const;
    QUALIFIER_DEVICE float GetArea() const { return area_; }

    QUALIFIER_DEVICE void SetPdfArea(float pdf_area) { pdf_area_ = pdf_area; }
    QUALIFIER_DEVICE void SetIdInstance(uint64_t id_instance) { id_instance_ = id_instance; }

private:
    QUALIFIER_DEVICE void IntersectTriangle(const Ray &ray, Bsdf **bsdf_buffer, Texture **texture_buffer,
                                            const float *pixel_buffer, uint64_t *seed,
                                            Intersection *its) const;
    QUALIFIER_DEVICE void IntersectSphere(const Ray &ray, Bsdf **bsdf_buffer, Texture **texture_buffer,
                                          const float *pixel_buffer, uint64_t *seed,
                                          Intersection *its) const;
    QUALIFIER_DEVICE void IntersectDisk(const Ray &ray, Bsdf **bsdf_buffer, Texture **texture_buffer,
                                          const float *pixel_buffer, uint64_t *seed,
                                          Intersection *its) const;

    Type type_;
    float area_;
    float pdf_area_;
    uint64_t id_bsdf_;
    uint64_t id_instance_;
    struct Triangle
    {
        Vec3 v0v1;
        Vec3 v0v2;
        Vec2 texcoords[3];
        Vec3 positions[3];
        Vec3 normals[3];
        Vec3 tangents[3];
        Vec3 bitangents[3];
    } geom_triangle_;
    struct Sphere
    {
        float radius;
        Vec3 center;
        Mat4 to_world;
        Mat4 to_local;
        Mat4 normal_to_world;
    } geom_sphere_;
    struct Disk
    {
        Mat4 to_world;
        Mat4 normal_to_world;
        Mat4 to_local;
    } geom_disk_;
};