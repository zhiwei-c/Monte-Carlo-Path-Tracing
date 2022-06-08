#pragma once

#include <string>

#include "intersection.h"

enum ShapeType
{
    kNoneShape,
    kMeshes,
    kRectangle,
    kCube,
    kSphere,
    kDisk
};

struct ShapeInfo
{
    ShapeType type;
    bool face_normals;
    bool flip_normals;
    bool flip_tex_coords;
    uint bsdf_idx;
    Float radius;
    vec3 center;
    std::string filename;
    gmat4 *to_world;

    ShapeInfo(const std::string &filename, bool face_normals, bool flip_normals, bool flip_tex_coords,
              gmat4 *to_world, uint bsdf_idx)
        : type(kMeshes), filename(filename), face_normals(face_normals), flip_normals(flip_normals),
          flip_tex_coords(flip_tex_coords), to_world(to_world), bsdf_idx(bsdf_idx)
    {
    }

    ShapeInfo(vec3 center, Float radius, bool flip_normals, gmat4 *to_world, uint bsdf_idx)
        : type(kSphere), center(center), radius(radius), flip_normals(flip_normals), to_world(to_world), bsdf_idx(bsdf_idx)
    {
    }

    ShapeInfo(ShapeType type, bool flip_normals, gmat4 *to_world, uint bsdf_idx)
        : type(type), flip_normals(flip_normals), to_world(to_world), bsdf_idx(bsdf_idx)
    {
    }

    ~ShapeInfo()
    {
        if (to_world)
        {
            delete to_world;
            to_world = nullptr;
        }
    }
};

class Vertex
{
public:
    vec2 texcoord;
    vec3 position;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;

    __host__ __device__ Vertex() : texcoord(vec2()), position(vec3()), normal(vec3()), tangent(vec3()), bitangent(vec3()) {}
};

class Shape
{
public:
    __device__ Shape() : flip_normals_(false), bsdf_(nullptr), pdf_area_(0) {}

    __device__ void SetOtherInfo(bool flip_normals, Float pdf_area)
    {
        pdf_area_ = pdf_area;
        flip_normals_ = flip_normals;
    }

    __device__ void Intersect(const Ray &ray, const vec2 &sample, Intersection &its) const;

    __device__ void SampleP(Intersection &its, const vec3 &sample) const;

    __device__ void InitTriangle(Vertex *v, Bsdf **bsdf, Float area);

private:
    bool flip_normals_;
    Vertex v_[3];
    Bsdf **bsdf_;
    Float area_;
    Float pdf_area_;
};

__global__ inline void CreateMeshes(uint max_x, uint max_y, uint mesh_num, Vertex *v_buffer, uvec3 *i_buffer, uint *m_idx,
                                    Bsdf **bsdfs, AABB *mesh_aabbs, Float *mesh_areas, Shape *mesh_list)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;

    auto idx = j * max_x + i;
    if (idx >= mesh_num)
        return;

    Vertex v[3];
    mesh_aabbs[idx] = AABB();
    for (int k = 0; k < 3; k++)
    {
        v[k] = v_buffer[i_buffer[idx][k]];
        mesh_aabbs[idx] += v[k].position;
    }

    mesh_areas[idx] = myvec::length(myvec::cross(v[1].position - v[0].position, v[2].position - v[0].position)) *
                      static_cast<Float>(0.5);
    mesh_list[idx].InitTriangle(v, bsdfs + m_idx[idx], mesh_areas[idx]);
}

__global__ inline void SetMeshesOtherInfo(uint meshes_idx_begin, uint meshes_num, bool flip_normals, Float shape_area,
                                          Shape *meshes_list_)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (uint i = 0; i < meshes_num; i++)
        {
            meshes_list_[meshes_idx_begin + i].SetOtherInfo(flip_normals, static_cast<Float>(1) / shape_area);
        }
    }
}
