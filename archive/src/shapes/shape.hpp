#pragma once

#include "../accelerators/aabb.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//几何形状的类型
enum class ShapeType
{
    //基本的几何形状
    kSphere,   //球面
    kDisk,     //圆盘
    kTriangle, //三角形
    kCylinder, //圆柱面

    //三角形的集合
    kRectangle, //矩形
    kCube,      //立方体
    kMeshes,    //网格模型
};

// 几何形状
class Shape
{
public:
    virtual ~Shape() {}

    virtual void Intersect(const Ray &ray, Sampler *sampler, Intersection *its) const = 0;
    virtual Intersection SamplePoint(Sampler *sampler) const = 0;

    double area() const { return area_; }
    AABB aabb() const { return aabb_; }

    void SetPdfArea(double pdf_area) { pdf_area_ = pdf_area; }
    virtual void SetBsdf(Bsdf *bsdf) { bsdf_ = bsdf; }
    virtual void SetMedium(Medium *medium_int, Medium *medium_ext) { medium_ext_ = medium_ext, medium_int_ = medium_int; }

protected:
    Shape(const std::string &id, ShapeType type, bool flip_normals)
        : id_(id),
          type_(type),
          area_(0),
          medium_int_(nullptr),
          medium_ext_(nullptr),
          bsdf_(nullptr),
          aabb_(AABB()),
          flip_normals_(flip_normals)
    {
    }

    bool flip_normals_;  //默认法线方向是否翻转
    Bsdf *bsdf_;         //表面材质
    Medium *medium_int_; //表面内侧介质
    Medium *medium_ext_; //表面外侧介质
    double area_;        //物体的表面积
    double pdf_area_;    //物体面元概率
    AABB aabb_;          //物体的轴对齐包围盒
    std::string id_;     //物体ID

private:
    ShapeType type_; //物体类型
};

NAMESPACE_END(raytracer)