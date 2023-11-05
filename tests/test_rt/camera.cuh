#pragma once

#include "ray_tracer.cuh"

using namespace rt;

class Camera
{
public:
    QUALIFIER_D_H Camera();
    QUALIFIER_D_H Camera(const uint32_t spp, const int width, const int height, const float fov_x,
                         const Vec3 eye, const Vec3 look_at, const Vec3 up);

    QUALIFIER_D_H int width() const { return width_; }
    QUALIFIER_D_H int height() const { return height_; }
    QUALIFIER_D_H uint32_t spp() const { return spp_; }
    QUALIFIER_D_H float spp_inv() const { return spp_inv_; }
    QUALIFIER_D_H float fov_x() const { return fov_x_; }
    QUALIFIER_D_H Vec3 eye() const { return eye_; }
    QUALIFIER_D_H Vec3 front() const { return front_; }
    QUALIFIER_D_H Vec3 view_dx() const { return view_dx_; }
    QUALIFIER_D_H Vec3 view_dy() const { return view_dy_; }

private:
    int width_;
    int height_;
    uint32_t spp_;
    float spp_inv_;
    float fov_x_;
    float fov_y_;
    Vec3 eye_;
    Vec3 front_;
    Vec3 right_;
    Vec3 up_;
    Vec3 view_dx_; // 当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，
                   // 在世界坐标系下移动的长度
    Vec3 view_dy_; // 当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，
                   // 在世界坐标系下移动的长度
};
