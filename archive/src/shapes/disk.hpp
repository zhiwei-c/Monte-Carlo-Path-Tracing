#pragma once

#include <vector>

#include "shape.hpp"
#include "../global.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"

NAMESPACE_BEGIN(raytracer)

//圆盘
class Disk : public Shape
{
public:
    Disk(const std::string &id, const dmat4 &to_world, bool flip_normals);

    void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

private:
    dmat4 to_local_;        //从世界坐标系到局部坐标系的变换矩阵
    dmat4 to_world_;        //从局部坐标系到世界坐标系的变换矩阵
    dmat4 normal_to_world_; //方向从局部坐标系到世界坐标系的变换矩阵
};

NAMESPACE_END(raytracer)