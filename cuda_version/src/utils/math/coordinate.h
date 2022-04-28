#pragma once

#include "../global.h"

__device__ inline bool NotSameHemis(const vec3 &a, const vec3 &b)
{
    return myvec::dot(a, b) < kEpsilon;
}

__device__ inline bool SameHemis(const vec3 &a, const vec3 &b)
{
    return myvec::dot(a, b) > -kEpsilon;
}

__device__ inline bool SameDirection(const vec3 &a, const vec3 &b)
{
    return abs(myvec::dot(a, b) - 1) < kEpsilonL;
}

__device__ inline bool Perpendicular(const vec3 &a, const vec3 &b)
{
    return abs(myvec::dot(a, b)) < kEpsilonL;
}

__host__ __device__ inline gvec3 TransfromDir(const gmat4 &trans, const gvec3 &dir)
{
    auto ret = gvec3(0);
    auto dir_tmp = trans * gvec4(dir, 0);
    for (int j = 0; j < 3; ++j)
        ret[j] = dir_tmp[j];
    return glm::normalize(ret);
}

inline gvec3 TransfromPt(const gmat4 &trans, const gvec3 &pt)
{
    auto ret = gvec3(0);
    auto pt_tmp = trans * gvec4(pt, 1);
    for (int j = 0; j < 3; ++j)
        ret[j] = pt_tmp[j] / pt_tmp.w;
    return ret;
}

__host__ __device__ inline vec3 ToWorld(const vec3 &dir, const vec3 &normal)
{
    auto B = vec3(0),
         C = vec3(0);
    if (abs(normal.x) > abs(normal.y))
    {
        auto len_inv = 1.0 / sqrt(normal.x * normal.x + normal.z * normal.z);
        C = vec3(normal.z * len_inv, 0, -normal.x * len_inv);
    }
    else
    {
        auto len_inv = 1.0 / sqrt(normal.y * normal.y + normal.z * normal.z);
        C = vec3(0, normal.z * len_inv, -normal.y * len_inv);
    }
    B = myvec::cross(C, normal);
    return myvec::normalize(dir.x * B + dir.y * C + dir.z * normal);
}

__host__ __device__ inline vec3 ToLocal(const vec3 &dir, const vec3 &up)
{
    auto B = vec3(0),
         C = vec3(0);
    if (abs(up.x) > abs(up.y))
    {
        auto len_inv = 1.0 / sqrt(up.x * up.x + up.z * up.z);
        C = vec3(up.z * len_inv, 0, -up.x * len_inv);
    }
    else
    {
        auto len_inv = 1.0 / sqrt(up.y * up.y + up.z * up.z);
        C = vec3(0, up.z * len_inv, -up.y * len_inv);
    }
    B = myvec::cross(C, up);
    return vec3(myvec::dot(dir, B), myvec::dot(dir, C), myvec::dot(dir, up));
}

#define UP_DIM_WORLD 1
#define FRONT_DIM_WORLD 2
#define RIGHT_DIM_WORLD 0

__device__ inline void CartesianToSpherical(const gvec3 &dir, Float &theta, Float &phi)
{
    auto cos_theta = dir[UP_DIM_WORLD];

    theta = acos(cos_theta);

    auto sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    auto cos_phi = dir[FRONT_DIM_WORLD] / sin_theta;
    cos_phi = glm::min(cos_phi, 1 - kEpsilon);
    cos_phi = glm::max(cos_phi, -1 + kEpsilon);

    auto sin_phi = dir[RIGHT_DIM_WORLD] / sin_theta;

    phi = (sin_phi > 0) ? acos(cos_phi) : (2.0 * kPi - acos(cos_phi));

    if (phi < 0)
        phi += 2 * kPi;

    if (phi > 2 * kPi)
        phi -= 2 * kPi;
}
