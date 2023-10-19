#pragma once

#include "../global.cuh"
#include "../accelerators/aabb.cuh"
#include "../renderer/ray.cuh"
#include "../renderer/intersection.cuh"

struct Vertex
{
    Vec2 texcoord;
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
};

class Primitive
{
public:
    enum Type
    {
        kInvalid,
        kTriangle,
        kSphere,
        kDisk,
    };

    struct Info
    {
        Type type;
        struct Data
        {
            uint32_t id_bsdf;

            struct Triangle
            {
                Vec2 texcoords[3];
                Vec3 positions[3];
                Vec3 normals[3];
                Vec3 tangents[3];
                Vec3 bitangents[3];
            } triangle;
            struct Sphere
            {
                float radius;
                Vec3 center;
                Mat4 to_world;
            } sphere;
            struct Disk
            {
                Mat4 to_world;
            } disk;
        } data;

        static Info CreateTriangle(Vertex *v, const uint32_t id_bsdf);
        static Info CreateSphere(const Vec3 &center, const float radius, const Mat4 &to_world,
                                 const uint32_t id_bsdf);
        static Info CreateDisk(const Mat4 &to_world, const uint32_t id_bsdf);
    };

    QUALIFIER_DEVICE Primitive();
    QUALIFIER_DEVICE Primitive(const uint32_t id_primitive, const Info &info);

    QUALIFIER_DEVICE void Intersect(const Ray &ray, Bsdf **bsdf_buffer, Texture **texture_buffer,
                                    const float *pixel_buffer, uint32_t *seed,
                                    Intersection *its) const;
    QUALIFIER_DEVICE void SamplePoint(uint32_t *seed, Intersection *its) const;

    QUALIFIER_DEVICE AABB aabb() const;
    QUALIFIER_DEVICE float area() const { return area_; }

    QUALIFIER_DEVICE void SetPdfArea(float pdf_area) { pdf_area_ = pdf_area; }
    QUALIFIER_DEVICE void SetIdInstance(uint32_t id_instance) { id_instance_ = id_instance; }

private:
    struct Geometry
    {
        struct Triangle
        {
            Vec3 v0v1;
            Vec3 v0v2;
            Vec2 texcoords[3];
            Vec3 positions[3];
            Vec3 normals[3];
            Vec3 tangents[3];
            Vec3 bitangents[3];
        } triangle;
        struct Sphere
        {
            float radius;
            Vec3 center;
            Mat4 to_world;
            Mat4 to_local;
            Mat4 normal_to_world;
        } sphere;
        struct Disk
        {
            Mat4 to_world;
            Mat4 normal_to_world;
            Mat4 to_local;
        } disk;
    };

    QUALIFIER_DEVICE void InitDisk(const Info::Data::Disk &data);
    QUALIFIER_DEVICE void InitSphere(const Info::Data::Sphere &data);
    QUALIFIER_DEVICE void InitTriangle(const Info::Data::Triangle &data);

    QUALIFIER_DEVICE void IntersectDisk(const Ray &ray, Bsdf **bsdf_buffer,
                                        Texture **texture_buffer,
                                        const float *pixel_buffer, uint32_t *seed,
                                        Intersection *its) const;

    QUALIFIER_DEVICE void IntersectSphere(const Ray &ray, Bsdf **bsdf_buffer,
                                          Texture **texture_buffer,
                                          const float *pixel_buffer, uint32_t *seed,
                                          Intersection *its) const;

    QUALIFIER_DEVICE void IntersectTriangle(const Ray &ray, Bsdf **bsdf_buffer,
                                            Texture **texture_buffer,
                                            const float *pixel_buffer, uint32_t *seed,
                                            Intersection *its) const;

    Type type_;
    float area_;
    float pdf_area_;
    uint32_t id_bsdf_;
    uint32_t id_primitive_;
    uint32_t id_instance_;
    Geometry geom_;
};