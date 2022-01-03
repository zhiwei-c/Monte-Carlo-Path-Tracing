#pragma once

#include <utility>
#include "../rendering/intersection.h"

NAMESPACE_BEGIN(simple_renderer)

//物体类型
enum class ShapeType
{
    kSphere,    //球体
    kDisk,      //圆
    kTriangle,  //三角面片
    kCube,      //标准立方体
    kRectangle, //标准矩形
    kMeshes     //三角面片聚合成的形体
};

//物体类
class Shape
{
public:
    /**
     * \brief 判断光线是否与物体相交，获取交点。
     * \param ray 待判断的光线
     * \return 交点（如果判断发现没有交点，则认为交点与光线起点间距离为无穷远）
     */
    virtual Intersection Intersect(const Ray &ray) const = 0;

    /**
     * \brief 在物体表面按表面积采样一点，
     * \param pos_pre 作为采样时起点的，当前光线与物体的交点
     * \return 由 Intersection 类型和 Float 类型构成的 pair，分别代表采样到的物体上的点，和采样到该点的概率。
     */
    virtual std::pair<Intersection, Float> SampleP(const Vector3 &pos_pre) const = 0;

    ///\return 物体包围盒
    AABB aabb() const { return aabb_; }

    ///\return 物体表面积
    Float area() const { return area_; }

    ///\return 物体是否发光
    bool HasEmission() const { return material_->HasEmission(); }

    ///\return 物体类型
    ShapeType type() const { return type_; }

protected:
    AABB aabb_;          //物体包围盒
    Float area_;         //物体表面积
    Material *material_; //材质
    bool flip_normals_;  //默认法线方向是否翻转

    Shape(ShapeType type, Material *material, bool flip_normals)
        : type_(type), material_(material), flip_normals_(flip_normals), aabb_(AABB()), area_(0) {}

private:
    ShapeType type_; //物体类型
};

NAMESPACE_END(simple_renderer)