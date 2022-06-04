#pragma once

#include "shape_info.h"

class Vertex
{
public:
    vec2 texcoord;
    vec3 position;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;

    __host__ __device__ Vertex()
        : texcoord(vec2()), position(vec3()), normal(vec3()), tangent(vec3()), bitangent(vec3()) {}
};

class Mesh
{
public:
    __device__ Mesh() : mesh_idx_in_shape_(kUintMax),
                        shape_idx_(kUintMax),
                        flip_normals_(false),
                        bsdf_(nullptr),
                        pdf_area_(0),
                        pre_(nullptr),
                        next_(nullptr) {}

    __device__ void InitTriangle(Vertex *v, Bsdf **bsdf, Float area, Mesh *pre, Mesh *next);

    __device__ void SetOtherInfo(uint mesh_idx_in_shape, uint shape_idx, bool flip_normals, Float pdf_area)
    {
        mesh_idx_in_shape_ = mesh_idx_in_shape;
        shape_idx_ = shape_idx;
        pdf_area_ = pdf_area;
        flip_normals_ = flip_normals;
    }

    __device__ void Intersect(const Ray &ray, const vec2 &sample, Intersection &its) const;

    __device__ void SampleP(Intersection &its, const vec3 &sample) const;

private:
    bool flip_normals_;
    uint mesh_idx_in_shape_;
    uint shape_idx_;
    Vertex v_[3];
    Bsdf **bsdf_;
    Float area_;
    Float pdf_area_;
    Mesh *pre_;
    Mesh *next_;
};