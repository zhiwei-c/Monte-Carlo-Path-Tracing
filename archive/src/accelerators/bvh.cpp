#include "bvh.hpp"

#include <algorithm>

#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../shapes/shape.hpp"

NAMESPACE_BEGIN(raytracer)

BvhNode::BvhNode(Shape *shape)
    : left(nullptr),
      right(nullptr),
      shape(shape),
      area(shape->area()),
      aabb(shape->aabb())
{
}

BvhNode::BvhNode(BvhNode *left, BvhNode *right)
    : left(left),
      right(right),
      shape(nullptr),
      area(left->area + right->area),
      aabb(left->aabb + right->aabb)
{
}

BvhNode::~BvhNode()
{
    if (left != nullptr)
    {
        delete left;
        left = nullptr;
    }
    if (right != nullptr)
    {
        delete right;
        right = nullptr;
    }
}

Bvh::Bvh(const std::vector<Shape *> &shapes)
    : Accelerator(),
      root_(BuildRecursively(shapes))
{
}

Bvh::~Bvh()
{
    delete root_;
    root_ = nullptr;
}

bool Bvh::Intersect(const Ray &ray, Sampler* sampler, Intersection *its) const
{
    BvhNode *node_stack[64] = {nullptr};
    node_stack[0] = root_;
    int top = 0;

    BvhNode *now = nullptr;
    while (top >= 0)
    {
        now = node_stack[top];
        --top;
        while (now->aabb.Intersect(ray))
        {
            if (now->shape != nullptr)
            {
                now->shape->Intersect(ray, sampler, its);
                break;
            }
            else
            {
                ++top;
                node_stack[top] = now->right;
                now = now->left;
            }
        }
    }
    return its->IsValid();
}

Intersection Bvh::SamplePoint(Sampler* sampler) const
{
    BvhNode *now = root_;
    while (now->shape == nullptr)
    {
        now = (sampler->Next1D() * now->area < now->left->area) ? now->left : now->right;
    }
    return now->shape->SamplePoint(sampler);
}

BvhNode *Bvh::BuildRecursively(std::vector<Shape *> shapes)
{
    if (shapes.empty())
    {
        return nullptr;
    }

    if (shapes.size() == 1)
    {
        return new BvhNode(shapes[0]);
    }

    auto aabb = AABB();
    for (Shape *shape : shapes)
    {
        aabb += shape->aabb();
    }

    double length_x = aabb.max().x - aabb.min().x,
          length_y = aabb.max().y - aabb.min().y,
          length_z = aabb.max().z - aabb.min().z;
    if (length_x > length_y && length_x > length_z)
    {
        std::sort(shapes.begin(), shapes.end(), [](Shape *a, Shape *b)
                  { return a->aabb().center().x < b->aabb().center().x; });
    }
    else if (length_y > length_z)
    {
        std::sort(shapes.begin(), shapes.end(), [](Shape *a, Shape *b)
                  { return a->aabb().center().y < b->aabb().center().y; });
    }
    else
    {
        std::sort(shapes.begin(), shapes.end(), [](Shape *a, Shape *b)
                  { return a->aabb().center().z < b->aabb().center().z; });
    }

    auto objs_left = std::vector<Shape *>(shapes.begin(), shapes.begin() + shapes.size() / 2),
         objs_right = std::vector<Shape *>(shapes.begin() + shapes.size() / 2, shapes.end());
    BvhNode *left = BuildRecursively(objs_left),
            *right = BuildRecursively(objs_right);
    return new BvhNode(left, right);
}

NAMESPACE_END(raytracer)