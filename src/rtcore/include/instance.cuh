#pragma once

#include <vector>

#include "bvh_builder.cuh"
#include "primitive.cuh"
#include "utils.cuh"

namespace csrt
{

class BLAS
{
public:
    QUALIFIER_D_H BLAS();
    QUALIFIER_D_H BLAS(const uint64_t offset_node, const BvhNode *node_buffer,
                       const uint64_t offset_primitive,
                       const Primitive *primitive_buffer);

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const Vec3 &xi) const;

private:
    const BvhNode *nodes_;
    const Primitive *primitives_;
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
            Mat4 to_world = {};
        };

        struct Sphere
        {
            float radius = 1.0f;
            Vec3 center = {};
            Mat4 to_world = {};
        };

        struct Rectangle
        {
            Mat4 to_world = {};
        };

        struct Meshes
        {
            std::vector<Vec2> texcoords = {};
            std::vector<Vec3> positions = {};
            std::vector<Vec3> normals = {};
            std::vector<Vec3> tangents = {};
            std::vector<Vec3> bitangents = {};
            std::vector<Uvec3> indices = {};
            Mat4 to_world = {};
        };

        Instance::Type type = Instance::Type::kNone;
        uint32_t id_bsdf = kInvalidId;
        Cube cube = {};
        Sphere sphere = {};
        Rectangle rectangle = {};
        Meshes meshes = {};

        static Instance::Info CreateCube(const Mat4 &to_world,
                                         const uint32_t id_bsdf);
        static Instance::Info CreateSphere(const float &radius,
                                           const Vec3 &center,
                                           const Mat4 &to_world,
                                           const uint32_t id_bsdf);
        static Instance::Info CreateRectangle(const Mat4 &to_world,
                                              const uint32_t id_bsdf);
    };

    QUALIFIER_D_H Instance();
    QUALIFIER_D_H Instance(const uint32_t id, const BLAS *blas_buffer);

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const Vec3 &xi) const;

private:
    uint32_t id_;
    const BLAS *blas_;
};

} // namespace csrt