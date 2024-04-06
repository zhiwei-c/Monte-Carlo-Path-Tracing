#pragma once

#include <vector>

#include "shape.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//球面
class Sphere : public Shape
{
public:
    Sphere(const std::string &id, const dvec3 &center, double radius, const dmat4 &to_world, bool flip_normals);

    void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

private:
    double radius_;         //球在局部空间的半径
    dvec3 center_;          //球在局部空间的中心
    dmat4 to_world_;        //从局部坐标系转换到世界坐标系的变换矩阵
    dmat4 noraml_to_world_; //方向从局部坐标系转换到世界坐标系的变换矩阵
    dmat4 to_local_;        //从世界坐标系转换到局部坐标系的变换矩阵
};

NAMESPACE_END(raytracer)