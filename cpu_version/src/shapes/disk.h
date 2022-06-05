#pragma once

#include "../core/shape_base.h"

NAMESPACE_BEGIN(raytracer)

class Disk : public Shape
{
public:
    /**
     * \brief 标准圆。在局部坐标下表示为：x^2 + y^2 <= 1，z=0
     * \param bsdf 材质
     * \param to_world 从局部坐标系到世界坐标系的变换矩阵
     * \param flip_normals 法线方向是否翻转
     */
    Disk(Bsdf *bsdf, Medium *int_medium, Medium *ext_medium, std::unique_ptr<Mat4> to_world, bool flip_normals);

    void Intersect(const Ray &ray, Intersection &its) const override;

    Intersection SampleP() const override;

private:
    Float area_inv_;
    std::unique_ptr<Mat4> to_world_; //从局部坐标系到世界坐标系的变换矩阵
    std::unique_ptr<Mat4> to_world_norm_;
    std::unique_ptr<Mat4> to_local_;
};

NAMESPACE_END(raytracer)