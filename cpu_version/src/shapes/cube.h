#pragma once

#include "triangle.h"
#include "../accelerator/bvh_accel.h"

NAMESPACE_BEGIN(raytracer)

class Cube : public Shape
{
public:
    /**
     * \brief 标准立方体。在局部坐标下表示为：-1 < x < 1，-1 < y < 1，-1 < z < 1
     * \param bsdf 材质
     * \param to_world 从局部坐标系到世界坐标系的变换矩阵
     * \param flip_normals 法线方向是否翻转
     */
    Cube(Bsdf *bsdf, Medium *int_medium, Medium *ext_medium, std::unique_ptr<Mat4> to_world, bool flip_normals);

    ~Cube()
    {
        for (auto &mesh : meshes_)
        {
            if (mesh)
            {
                delete mesh;
                mesh = nullptr;
            }
        }
    }

    void Intersect(const Ray &ray, Intersection &its) const override
    {
        this->bvh_->Intersect(ray, its);
    }

    Intersection SampleP() const override
    {
        return this->bvh_->Sample();
    }

private:
    std::unique_ptr<BvhAccel> bvh_; //层次包围盒
    std::vector<Shape *> meshes_;   //包含的三角面片
};

NAMESPACE_END(raytracer)