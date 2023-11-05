#pragma once

#include <vector>

#include "bvh_builder.cuh"
#include "primitive.cuh"

namespace rt
{

class BLAS
{
public:
    QUALIFIER_D_H BLAS();
    QUALIFIER_D_H BLAS(BvhNode *nodes, Primitive *primitives);

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1, const float xi_2) const;

private:
    BvhNode *nodes_;
    Primitive *primitives_;
};

class Instance
{
public:
    enum class Type
    {
        kNone,
        kCube,
        kSphere,
        kRectangle,
        kMeshes,
    };

    struct Info
    {
        struct Cube
        {
            Mat4 to_world;
        };

        struct Sphere
        {
            float radius;
            Vec3 center;
            Mat4 to_world;
        };

        struct Rectangle
        {
            Mat4 to_world;
        };

        struct Meshes
        {
            std::vector<Vec2> texcoords;
            std::vector<Vec3> positions;
            std::vector<Vec3> normals;
            std::vector<Vec3> tangents;
            std::vector<Vec3> bitangents;
            std::vector<Uvec3> indices;
            Mat4 to_world;
        };

        Instance::Type type;
        union
        {
            Cube cube;
            Sphere sphere;
            Rectangle rectangle;
            Meshes meshes;
        };

        Info();
        ~Info() {}
        Info(const Instance::Info &info);
        void operator=(const Instance::Info &info);
    };

    QUALIFIER_D_H Instance();
    QUALIFIER_D_H Instance(const uint32_t id_instance, BLAS *accel);

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1, const float xi_2) const;

private:
    uint32_t id_instance_;
    BLAS *accel_;
};

} // namespace rt