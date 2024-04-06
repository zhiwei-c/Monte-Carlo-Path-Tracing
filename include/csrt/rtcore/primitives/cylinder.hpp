#ifndef CSRT__RTCORE__PRIMITIVES_CYLINDER_HPP
#define CSRT__RTCORE__PRIMITIVES_CYLINDER_HPP

#include "../accel/aabb.hpp"
#include "../hit.hpp"

namespace csrt
{

class Bsdf;

struct CylinderInfo
{
    float radius = 1.0f; // 在局部坐标系下横截面半径
    Vec3 p0 = {};        // 在局部坐标系下底部中心
    Vec3 p1 = {};        // 在局部坐标系下顶部中心
    Mat4 to_world = {};  // 从局部坐标系到世界坐标系的变换矩阵
};

struct CylinderData
{
    // 在世界坐标系下横截面半径
    float radius = 1.0f;
    // 在世界坐标系下高
    float length = 1.0f;
    // 圆柱面从局部坐标系（此时0<=z<=length, x^2+y^2=radius）变换到世界坐标系的变换矩阵
    Mat4 to_world = {};
};

QUALIFIER_D_H AABB GetAabbCylinder(const CylinderData &data);

QUALIFIER_D_H bool IntersectCylinder(const uint32_t id_primitive,
                                     const CylinderData &data, Bsdf *bsdf,
                                     uint32_t *seed, Ray *ray, Hit *hit);

QUALIFIER_D_H Hit SampleCylinder(const uint32_t id_primitive,
                                 const CylinderData &data, const float xi_0,
                                 const float xi_1);

} // namespace csrt

#endif