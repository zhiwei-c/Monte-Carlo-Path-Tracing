#ifndef CSRT__RTCORE__INSTANCE_HPP
#define CSRT__RTCORE__INSTANCE_HPP

#include <vector>

#include "../utils.hpp"
#include "accel/blas.hpp"
#include "primitives/primitive.hpp"

namespace csrt
{

enum class InstanceType
{
    kNone,
    kCube,
    kSphere,
    kRectangle,
    kMeshes,
};

struct SphereInfo
{
    float radius = 1.0f;
    Vec3 center = {};
};

struct MeshesInfo
{
    std::vector<Vec2> texcoords = {};
    std::vector<Vec3> positions = {};
    std::vector<Vec3> normals = {};
    std::vector<Vec3> tangents = {};
    std::vector<Vec3> bitangents = {};
    std::vector<Uvec3> indices = {};
};

struct InstanceInfo
{
    InstanceType type = InstanceType::kNone;
    uint32_t id_bsdf = kInvalidId;
    uint32_t id_medium_int = kInvalidId;
    uint32_t id_medium_ext = kInvalidId;
    bool flip_normals = false;
    Mat4 to_world = {};
    SphereInfo sphere = {};
    MeshesInfo meshes = {};
};

class Instance
{
public:
    QUALIFIER_D_H Instance();
    QUALIFIER_D_H Instance(const uint32_t id, const uint32_t id_medium_int,
                           const uint32_t id_medium_ext,
                           const BLAS *blas_buffer);

    QUALIFIER_D_H void Intersect(Bsdf *bsdf_buffer, uint32_t *map_instance_bsdf,
                                 uint32_t *seed, Ray *ray, Hit *hit) const;

    QUALIFIER_D_H bool IntersectAny(Bsdf *bsdf_buffer,
                                    uint32_t *map_instance_bsdf, uint32_t *seed,
                                    Ray *ray) const;

    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1,
                             const float xi_2) const;

private:
    uint32_t id_;
    uint32_t id_medium_int_;
    uint32_t id_medium_ext_;
    const BLAS *blas_;
};

} // namespace csrt

#endif