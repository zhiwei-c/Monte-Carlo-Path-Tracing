#pragma once

#include "math_base.h"

NAMESPACE_BEGIN(raytracer)

constexpr int UP_DIM_WORLD = 1;
constexpr int FRONT_DIM_WORLD = 2;
constexpr int RIGHT_DIM_WORLD = 0;

///\brief 对位置左乘齐次坐标矩阵，进行变换
inline Vector3 TransfromDir(Mat4 trans, Vector3 dir)
{
    Vector3 ret(0);
    auto dir_tmp = trans * glm::vec4(dir, 0);
    for (int j = 0; j < 3; ++j)
        ret[j] = dir_tmp[j];
    return glm::normalize(ret);
}

///\brief 对方向左乘齐次坐标矩阵，进行变换
inline Vector3 TransfromPt(Mat4 trans, Vector3 pt)
{
    Vector3 ret(0);
    auto pt_tmp = trans * glm::vec4(pt, 1);
    for (int j = 0; j < 3; ++j)
        ret[j] = pt_tmp[j] / pt_tmp.w;
    return ret;
}

///\brief 两个向量是否同向
inline bool SameDirection(const Vector3 &a, const Vector3 &b)
{
    return FloatEqual(glm::dot(a, b), 1, kEpsilonPdf);
}

///\brief 两个向量是否垂直
inline bool Perpendicular(const Vector3 &a, const Vector3 &b)
{
    return FloatEqual(glm::dot(a, b), 0, kEpsilonPdf);
}

///\brief 两个向量是否不在同一个半球之中（夹角大于90度）
inline bool NotSameHemis(const Vector3 &a, const Vector3 &b)
{
    return glm::dot(a, b) < 0;
}

///\brief 两个向量是否在同一个半球之中（夹角小于90度）
inline bool SameHemis(const Vector3 &a, const Vector3 &b)
{
    return glm::dot(a, b) > 0;
}

///\brief 将单位向量从局部坐标系转换到世界坐标系
///\param dir 待转换的单位向量
///\param up 局部坐标系的竖直向上方向在世界坐标系下的方向
inline Vector3 ToWorld(const Vector3 &dir, const Vector3 &up)
{
    auto B = Vector3(0), C = Vector3(0);
    if (std::fabs(up.x) > std::fabs(up.y))
    {
        Float len_inv = 1 / std::sqrt(up.x * up.x + up.z * up.z);
        C = Vector3(up.z * len_inv, 0, -up.x * len_inv);
    }
    else
    {
        Float len_inv = 1 / std::sqrt(up.y * up.y + up.z * up.z);
        C = Vector3(0, up.z * len_inv, -up.y * len_inv);
    }
    B = glm::cross(C, up);
    return glm::normalize(dir.x * B + dir.y * C + dir.z * up);
}

///\brief 将单位向量从世界坐标系转换到局部坐标系
///\param dir 待转换的单位向量
///\param up 局部坐标系的竖直向上方向在世界坐标系下的方向
inline Vector3 ToLocal(const Vector3 &dir, const Vector3 &up)
{
    auto B = Vector3(0), C = Vector3(0);
    if (std::fabs(up.x) > std::fabs(up.y))
    {
        Float len_inv = 1 / std::sqrt(up.x * up.x + up.z * up.z);
        C = Vector3(up.z * len_inv, 0, -up.x * len_inv);
    }
    else
    {
        Float len_inv = 1.0f / std::sqrt(up.y * up.y + up.z * up.z);
        C = Vector3(0, up.z * len_inv, -up.y * len_inv);
    }
    B = glm::cross(C, up);
    return Vector3(glm::dot(dir, B), glm::dot(dir, C), glm::dot(dir, up));
}

///\brief 将单位向量坐标从笛卡尔坐标系转换到球坐标系
///\param dir - 待转换的单位向量
///\param theta - 向量与 Up 方向的夹角（天顶角）
///\param phi - 向量与 Front 方向的夹角（方位角）
inline void CartesianToSpherical(const Vector3 &dir, Float &theta, Float &phi)
{
    Float cos_theta = dir[UP_DIM_WORLD];
    theta = glm::acos(cos_theta);
    Float sin_theta = std::sqrt(1 - cos_theta * cos_theta),
          cos_phi = Clamp<Float>(-1 + kEpsilon, 1 - kEpsilon, dir[FRONT_DIM_WORLD] / sin_theta),
          sin_phi = dir[RIGHT_DIM_WORLD] / sin_theta;
    phi = (sin_phi > 0) ? glm::acos(cos_phi) : (2 * kPi - glm::acos(cos_phi));
    if (phi < 0)
        phi += 2 * kPi;
    if (phi > 2 * kPi)
        phi -= 2 * kPi;
}

///\brief 将起点是原点的向量从笛卡尔坐标系转换到球坐标系
///\param dir 待转换的，起点是原点的向量
///\param phi 向量与 Front 方向的夹角（方位角）
///\param theta 向量与 Up 方向的夹角（天顶角）
///\param r 向量长度
inline void CartesianToSpherical(const Vector3 &dir, Float &theta, Float &phi, Float &r)
{
    r = glm::length(dir);
    CartesianToSpherical(glm::normalize(dir), theta, phi);
}

///\brief 将单位向量坐标从球坐标系转换到笛卡尔坐标系
///\param phi 向量在球坐标系下的方位角
///\param theta 向量在球坐标系下的天顶角
///\return 转换得到的单位向量
inline Vector3 SphericalToCartesian(Float theta, Float phi, Float r = 1)
{
    auto dir = Vector3(0);
    Float sin_theta = std::sin(theta);
    dir[UP_DIM_WORLD] = r * std::cos(theta);
    dir[RIGHT_DIM_WORLD] = r * std::sin(phi) * sin_theta;
    dir[FRONT_DIM_WORLD] = r * std::cos(phi) * sin_theta;
    return dir;
}

NAMESPACE_END(raytracer)