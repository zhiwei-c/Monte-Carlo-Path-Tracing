#pragma once

#include <vector>

#include "../utils.cuh"
#include "accel/blas.cuh"
#include "primitives/primitive.cuh"

namespace csrt
{

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

    QUALIFIER_D_H void Intersect(Bsdf *bsdf_buffer, uint32_t *map_instance_bsdf,
                                 uint32_t *seed, Ray *ray, Hit *hit) const;

    QUALIFIER_D_H bool IntersectAny(Bsdf *bsdf_buffer,
                                    uint32_t *map_instance_bsdf, uint32_t *seed,
                                    Ray *ray) const;

    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1,
                             const float xi_2) const;

private:
    uint32_t id_;
    const BLAS *blas_;
};

} // namespace csrt