#pragma once

#include <vector>

#include "shape.hpp"
#include "../global.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"

NAMESPACE_BEGIN(raytracer)

//正方形面片，由两个三角形面片构成
class Rectangle : public Shape
{
public:
    Rectangle(const std::string &id, const dmat4 &to_world, bool flip_normals);
    ~Rectangle();

    void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

    void SetBsdf(Bsdf *bsdf) override;
    void SetMedium(Medium *medium_int, Medium *medium_ext) override;

private:
    std::array<Shape *, 2> meshes_; //包含的三角面片
};

NAMESPACE_END(raytracer)