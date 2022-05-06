#pragma once

#include "../utils/global.h"
#include "integrator.h"

struct CameraInfo
{
    int height;
    int width;
    int sample_count;
    Float gamma;
    Float fov_height;
    vec3 eye_pos;
    vec3 look_dir;
    vec3 up;
};

class Camera
{
public:
    __device__ Camera() {}

    __device__ void InitCamera(const CameraInfo &info)
    {
        info_ = info;
        auto fov_width = info_.fov_height * info_.width / info_.height;
        auto right_dir = myvec::normalize(myvec::cross(info_.look_dir, info_.up));
        info_.up = myvec::normalize(myvec::cross(right_dir, info_.look_dir));
        view_dx_ = right_dir * static_cast<Float>(glm::tan(glm::radians(0.5 * fov_width)));
        view_dy_ = info_.up * static_cast<Float>(glm::tan(glm::radians(0.5 * info_.fov_height)));
    }

    __device__ vec3 GetDirection(int u, int v, const vec2 &offset)
    {
        auto x = static_cast<Float>(2 * (u + offset.x) / info_.width - 1);
        auto y = static_cast<Float>(1 - 2 * (v + offset.y) / info_.height);
        auto raw = info_.look_dir + x * view_dx_ + y * view_dy_;
        auto dir = myvec::normalize(raw);
        return dir;
    }

    __device__ vec3 EyePosition() const { return info_.eye_pos; }

    __device__ int SampleCount() const { return info_.sample_count; }

    __device__ Float GammaInv() const { return 1.0 / info_.gamma; }

private:
    CameraInfo info_;
    vec3 view_dx_; //当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，在世界坐标系下移动的长度
    vec3 view_dy_; //当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，在世界坐标系下移动的长度
};

__global__ void InitCamera(CameraInfo info, Camera *camera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        camera->InitCamera(info);
    }
}