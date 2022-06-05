#pragma once

#include "../core/shape_base.h"

NAMESPACE_BEGIN(raytracer)

class Sphere : public Shape
{
public:
    /**
     * \brief 球
     * \param bsdf 材质
     * \param center 球心
     * \param radius 半径
     * \param to_world 从局部坐标系到世界坐标系的变换矩阵
     * \param flip_normals 法线方向是否翻转
     */
    Sphere(Bsdf *bsdf, Medium *medium, const Vector3 &center, Float radius, std::unique_ptr<Mat4> to_world, bool flip_normals);

    void Intersect(const Ray &ray, Intersection &its) const override;

    Intersection SampleP() const override;

private:
    Float radius_;                        //半径
    Vector3 center_;                      //球心
    std::unique_ptr<Mat4> to_world_;      //从局部坐标系到世界坐标系位置的变换矩阵
    std::unique_ptr<Mat4> to_world_norm_; //从局部坐标系到世界坐标系方向的变换矩阵
    std::unique_ptr<Mat4> to_local_;      //从世界坐标系到局部坐标系位置的变换矩阵
};

NAMESPACE_END(raytracer)