#ifndef CSRT__RENDERER__CAMERA_HPP
#define CSRT__RENDERER__CAMERA_HPP

#include "../tensor.hpp"

namespace csrt
{

class Camera
{
public:
    struct Info
    {
        uint32_t spp = 64;
        int width = 1024;
        int height = 1024;
        float fov_x = 19.5;
        Vec3 eye = {0.0f, 1.0f, 6.8f};
        Vec3 look_at = {0.0f, 1.0f, 0.0f};
        Vec3 up = {0.0f, 1.0f, 0.0f};
    };

    QUALIFIER_D_H Camera();
    QUALIFIER_D_H Camera(const Camera::Info &info);

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
    // 当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，在世界坐标系下移动的向量
    Vec3 view_dx_;
    // 当光线于视锥体底的投影点在相机坐标系向x方向移动单位长度时，在世界坐标系下移动的向量
    Vec3 view_dy_;
};

} // namespace csrt

#endif