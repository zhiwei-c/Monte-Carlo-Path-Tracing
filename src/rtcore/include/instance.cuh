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
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1,
                             const float xi_2) const;

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
        struct Sphere
        {
            float radius = 1.0f;
            Vec3 center = {};
        };

        struct Meshes
        {
            std::vector<Vec2> texcoords = {};
            std::vector<Vec3> positions = {};
            std::vector<Vec3> normals = {};
            std::vector<Vec3> tangents = {};
            std::vector<Vec3> bitangents = {};
            std::vector<Uvec3> indices = {};
        };

        Instance::Type type = Instance::Type::kNone;
        uint32_t id_bsdf = kInvalidId;
        bool flip_normals = false;
        Mat4 to_world = {};
        Sphere sphere = {};
        Meshes meshes = {};
    };

    QUALIFIER_D_H Instance();
    QUALIFIER_D_H Instance(const uint32_t id, const BLAS *blas_buffer);

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1,
                             const float xi_2) const;

private:
    uint32_t id_;
    const BLAS *blas_;
};

} // namespace csrt