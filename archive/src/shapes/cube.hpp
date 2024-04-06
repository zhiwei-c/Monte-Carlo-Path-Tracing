#pragma once

#include <vector>

#include "shape.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//立方体，由十二个三角形面片构成
class Cube : public Shape
{
public:
    Cube(const std::string &id, const dmat4 &to_world, bool flip_normals);
    ~Cube();

    void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

    void SetBsdf(Bsdf *bsdf) override;
    void SetMedium(Medium *medium_int, Medium *medium_ext) override;

private:
    std::array<Shape *, 12> meshes_; //包含的三角面片
};

NAMESPACE_END(raytracer)