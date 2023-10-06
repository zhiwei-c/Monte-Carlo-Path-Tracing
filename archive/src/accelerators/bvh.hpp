#pragma once

#include "../global.hpp"

#include "aabb.hpp"
#include "accelerator.hpp"

NAMESPACE_BEGIN(raytracer)

//层次包围盒的节点
struct BvhNode
{
    double area;           //节点包含景物的总表面积
    Shape *shape;          //节点包含的物体。仅在叶节点非空。
    BvhNode *left, *right; //子节点
    AABB aabb;             //节点的轴对齐包围盒

    BvhNode(Shape *shape);
    BvhNode(BvhNode *left, BvhNode *right);
    ~BvhNode();
};

//层次包围盒（bounding volume hierarchy）
class Bvh : public Accelerator
{
public:
    Bvh(const std::vector<Shape *> &shapes);
    ~Bvh();

    bool Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

    double area() const override { return root_->area; }
    AABB aabb() const override { return root_->aabb; }

private:
    BvhNode *BuildRecursively(std::vector<Shape *> shapes);

    BvhNode *root_; //层次包围盒根节点
};

NAMESPACE_END(raytracer)