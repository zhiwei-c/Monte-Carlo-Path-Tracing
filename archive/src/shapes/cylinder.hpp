#pragma once

#include <vector>

#include "shape.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//球面
class Cylinder : public Shape
{
public:
    Cylinder(const std::string &id, const dvec3 &p0, const dvec3 &p1, double radius, const dmat4 &to_world, bool flip_normals);

    void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

private:
    double radius_;         //圆柱的半径
    double length_;         //圆柱母线的长度
    dmat4 to_world_;        //从局部坐标系转换到世界坐标系的变换矩阵
    dmat4 noraml_to_world_; //方向从局部坐标系转换到世界坐标系的变换矩阵
    dmat4 to_local_;        //从世界坐标系转换到局部坐标系的变换矩阵
};

NAMESPACE_END(raytracer)