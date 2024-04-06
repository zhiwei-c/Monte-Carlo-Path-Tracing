#pragma once

#include <vector>

#include "shape.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//三角形面片
class Triangle : public Shape
{
public:
    Triangle(const std::string &id, const std::vector<dvec3> &positions, const std::vector<dvec3> &normals,
             const std::vector<dvec3> &tangents, const std::vector<dvec3> &bitangents, const std::vector<dvec2> &texcoords,
             bool flip_normals);

    void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const override;
    Intersection SamplePoint(Sampler *sampler) const override;

private:
    dvec3 v0v1_;                    //三角形的边
    dvec3 v0v2_;                    //三角形的边
    std::vector<dvec2> texcoords_;  //顶点纹理坐标
    std::vector<dvec3> positions_;  //顶点世界空间坐标
    std::vector<dvec3> normals_;    //顶点世界空间法线方向
    std::vector<dvec3> tangents_;   //顶点世界空间切线方向
    std::vector<dvec3> bitangents_; //顶点世界空间副切线方向
};

NAMESPACE_END(raytracer)