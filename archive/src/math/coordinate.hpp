#pragma once

#include "math.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

constexpr int UP_DIM_WORLD = 1;
constexpr int FRONT_DIM_WORLD = 2;
constexpr int RIGHT_DIM_WORLD = 0;

///\brief 对方向左乘齐次坐标矩阵，进行变换
inline dvec3 TransfromPoint(dmat4 trans, dvec3 pt)
{
    dvec4 pt_tmp = trans * dvec4(pt, 1);
    return {pt_tmp.x / pt_tmp.w, pt_tmp.y / pt_tmp.w, pt_tmp.z / pt_tmp.w};
}

///\brief 对方向左乘齐次坐标矩阵，进行变换
inline dvec2 TransfromPoint(dmat3 trans, dvec2 pt)
{
    dvec3 pt_tmp = trans * dvec3(pt, 1);
    return {pt_tmp.x / pt_tmp.z, pt_tmp.y / pt_tmp.z};
}

///\brief 对位置左乘齐次坐标矩阵，进行变换
inline dvec3 TransfromVec(dmat4 trans, dvec3 dir)
{
    dvec4 dir_tmp = trans * dvec4(dir, 0.0);
    return glm::normalize(dvec3{dir_tmp.x, dir_tmp.y, dir_tmp.z});
}

///\brief 判断两个向量的方向是否相同
inline bool SameDirection(const dvec3 &a, const dvec3 &b)
{
    return std::abs(glm::dot(a, b) - 1.0) < 0.1;
}

inline void CoordinateSystem(const dvec3 &up, dvec3 &B, dvec3 &C)
{
    if (std::fabs(up.x) > std::fabs(up.y))
    {
        double len_inv = 1.0 / std::sqrt(up.x * up.x + up.z * up.z);
        C = dvec3(up.z * len_inv, 0, -up.x * len_inv);
    }
    else
    {
        double len_inv = 1.0 / std::sqrt(up.y * up.y + up.z * up.z);
        C = dvec3(0, up.z * len_inv, -up.y * len_inv);
    }
    B = glm::cross(C, up);
}

///\brief 将单位向量从局部坐标系转换到世界坐标系
///\param dir 待转换的单位向量
///\param up 局部坐标系的竖直向上方向在世界坐标系下的方向
inline dvec3 ToWorld(const dvec3 &dir, const dvec3 &up)
{
    auto B = dvec3(0), C = dvec3(0);
    CoordinateSystem(up, B, C);
    return glm::normalize(dir.x * B + dir.y * C + dir.z * up);
}

inline dmat4 ToWorld(const dvec3 &up)
{
    auto B = dvec3(0), C = dvec3(0);
    CoordinateSystem(up, B, C);
    return dmat4{B.x, B.y, B.z, 0,
                 C.x, C.y, C.z, 0,
                 up.x, up.y, up.z, 0,
                 0, 0, 0, 1};
}

///\brief 将单位向量从世界坐标系转换到局部坐标系
///\param dir 待转换的单位向量
///\param up 局部坐标系的竖直向上方向在世界坐标系下的方向
inline dvec3 ToLocal(const dvec3 &dir, const dvec3 &up)
{
    auto B = dvec3(0), C = dvec3(0);
    if (std::fabs(up.x) > std::fabs(up.y))
    {
        double len_inv = 1.0 / std::sqrt(up.x * up.x + up.z * up.z);
        C = dvec3(up.z * len_inv, 0, -up.x * len_inv);
    }
    else
    {
        double len_inv = 1.0 / std::sqrt(up.y * up.y + up.z * up.z);
        C = dvec3(0, up.z * len_inv, -up.y * len_inv);
    }
    B = glm::cross(C, up);
    return dvec3(glm::dot(dir, B), glm::dot(dir, C), glm::dot(dir, up));
}

///\brief 将向量从笛卡尔坐标系转换到球坐标系
///\param dir - 待转换的单位向量
///\param theta - 向量与 Up 方向的夹角（天顶角）
///\param phi - 向量与 Front 方向的夹角（方位角）
///\param r - 向量的长度
inline void CartesianToSpherical(dvec3 dir, double *theta, double *phi, double *r = nullptr)
{
    if (r != nullptr)
    {
        *r = glm::length(dir);
    }
    dir = glm::normalize(dir);

    const double cos_theta = dir[UP_DIM_WORLD];
    *theta = glm::acos(cos_theta);

    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta),
           cos_phi = dir[FRONT_DIM_WORLD] / sin_theta,
           sin_phi = dir[RIGHT_DIM_WORLD] / sin_theta;
    cos_phi = std::min(1.0, std::max(-1.0, cos_phi));
    *phi = (sin_phi > 0) ? glm::acos(cos_phi) : (2.0 * kPi - glm::acos(cos_phi));
    if (*phi < 0.0)
    {
        *phi += 2.0 * kPi;
    }
    if (*phi > 2.0 * kPi)
    {
        *phi -= 2.0 * kPi;
    }
}

///\brief 将向量从球坐标系转换到笛卡尔坐标系
///\param phi 向量在球坐标系下的方位角
///\param theta 向量在球坐标系下的天顶角
///\param r 向量在球坐标系下的长度
///\return 转换得到的向量
inline dvec3 SphericalToCartesian(double theta, double phi, double r = 1)
{
    auto dir = dvec3(0);
    double sin_theta = std::sin(theta);
    dir[UP_DIM_WORLD] = r * std::cos(theta);
    dir[RIGHT_DIM_WORLD] = r * std::sin(phi) * sin_theta;
    dir[FRONT_DIM_WORLD] = r * std::cos(phi) * sin_theta;
    return dir;
}

NAMESPACE_END(raytracer)