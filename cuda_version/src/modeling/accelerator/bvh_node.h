#pragma once

#include "../shape.h"

struct BvhNodeInfo
{
    bool valid;
    bool leaf;
    uint idx;
    uint left_idx;
    uint right_idx;
    AABB aabb;
    uint obj_idx;
    Float area;

    __host__ __device__ BvhNodeInfo()
        : valid(false),
          leaf(true),
          idx(kUintMax),
          left_idx(kUintMax),
          right_idx(kUintMax),
          aabb(AABB()),
          obj_idx(kUintMax),
          area(0) {}

    BvhNodeInfo(bool leaf,
                uint idx,
                uint left_idx,
                uint right_idx,
                const AABB &aabb,
                uint obj_idx,
                Float area)
        : valid(true),
          leaf(leaf),
          idx(idx),
          left_idx(left_idx),
          right_idx(right_idx),
          aabb(aabb),
          obj_idx(obj_idx),
          area(area) {}
};

class BvhNode
{
public:
    __device__ BvhNode()
        : leaf_(true), aabb_(AABB()), mesh_(nullptr), left_(nullptr), right_(nullptr), area_(0) {}

    __device__ void InitBvhNode(const AABB &aabb, Float area, BvhNode *left, BvhNode *right)
    {
        leaf_ = false;
        aabb_ = aabb;
        area_ = area;
        mesh_ = nullptr;
        left_ = left;
        right_ = right;
    }

    __device__ void InitBvhNode(const AABB &aabb, Mesh *mesh, Float area)
    {
        leaf_ = true;
        aabb_ = aabb;
        area_ = area;
        mesh_ = mesh;
        left_ = nullptr;
        right_ = nullptr;
    }

    __device__ bool IntersectAabb(const Ray &ray)
    {
        return aabb_.Intersect(ray);
    }

    __device__ void Intersect(const Ray &ray, const vec2 &sample, Intersection &its) const
    {
        mesh_->Intersect(ray, sample, its);
    }

    __device__ void SampleP(Intersection &its, const vec3 &sample) const
    {
        mesh_->SampleP(its, sample);
    }

    __device__ bool Leaf() const
    {
        return leaf_;
    }

    __device__ Float area() const
    {
        return area_;
    }

    __device__ AABB aabb() const
    {
        return aabb_;
    }

    __device__ BvhNode *left() const
    {
        return left_;
    }

    __device__ BvhNode *right() const
    {
        return right_;
    }

private:
    bool leaf_;
    AABB aabb_;
    Mesh *mesh_;
    Float area_;
    BvhNode *left_;
    BvhNode *right_;
};
