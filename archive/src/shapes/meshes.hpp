#pragma once

#include <vector>

#include "shape.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//网格模型
class Meshes : public Shape
{
public:
    Meshes(const std::string &id, const std::vector<Shape *> &meshes, bool flip_normals);
    ~Meshes();

    void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

    void SetBsdf(Bsdf *bsdf) override;
    void SetMedium(Medium *medium_int, Medium *medium_ext) override;

private:
    Accelerator *accelerator_;    //加速光线求交的数据结构
    std::vector<Shape *> meshes_; //包含的面片
};

NAMESPACE_END(raytracer)