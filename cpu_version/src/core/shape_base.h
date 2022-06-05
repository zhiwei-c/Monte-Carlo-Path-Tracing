#pragma once

#include "intersection.h"

NAMESPACE_BEGIN(raytracer)

//物体类型
enum class ShapeType
{
    kSphere,    //球
    kDisk,      //圆盘
    kTriangle,  //三角形
    kCube,      //立方体
    kRectangle, //矩形
    kMeshes     //网格模型
};

//物体基类
class Shape
{
public:
    virtual ~Shape() {}

    ///\brief 求取光线与物体的交点
    virtual void Intersect(const Ray &ray, Intersection &its) const = 0;

    ///\brief 按表面积从物体表面采样一点
    virtual Intersection SampleP() const = 0;

    ///\return 物体的轴对齐包围盒
    AABB aabb() const { return aabb_; }

    ///\return 物体是否发光
    bool HasEmission() const { return bsdf_->HasEmission(); }

    ///\return 物体的表面积
    Float area() const { return area_; }

    ///\brief 设置面元相应的概率
    void SetPdfArea(Float pdf_area) { pdf_area_ = pdf_area; }

protected:
    ///\brief 物体基类
    ///\param type 物体类型
    ///\param bsdf 材质
    ///\param flip_normals 默认法线方向是否翻转
    Shape(ShapeType type, Bsdf *bsdf, Medium *medium, bool flip_normals)
        : type_(type), bsdf_(bsdf), medium_(medium), flip_normals_(flip_normals), aabb_(AABB()), area_(0)
    {
    }

    bool flip_normals_; //默认法线方向是否翻转
    Float area_;        //物体的表面积
    Float pdf_area_;    //面元概率
    AABB aabb_;         //物体的轴对齐包围盒
    Bsdf *bsdf_;        //表面材质
    Medium *medium_;    //内部介质

private:
    ShapeType type_; //物体类型
};

NAMESPACE_END(raytracer)