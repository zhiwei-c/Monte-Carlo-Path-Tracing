#include "camera.cuh"

#include <cmath>

QUALIFIER_D_H Camera::Camera()
    : spp_(4), width_(512), height_(512), fov_x_(19.5f), eye_{0.0f, 1.0f, 6.8f},
      up_{0.0f, 1.0f, 0.0f}
{
    const Vec3 look_at = {0.0f, 1.0f, 0.0f};
    fov_y_ = fov_x_ * height_ / width_;
    front_ = Normalize(look_at - eye_);
    right_ = Normalize(Cross(front_, up_));
    up_ = Normalize(Cross(right_, front_));

    view_dx_ = right_ * tanf(ToRadians(0.5f * fov_x_));
    view_dy_ = up_ * tanf(ToRadians(0.5f * fov_y_));

    spp_inv_ = 1.0f / spp_;
}

QUALIFIER_D_H Camera::Camera(const uint32_t spp, const int width, const int height,
                             const float fov_x, const Vec3 eye, const Vec3 look_at, const Vec3 up)
    : spp_(spp), width_(width), height_(height), fov_x_(fov_x), eye_(eye), up_(up)
{
    fov_y_ = fov_x * height / width;
    front_ = Normalize(look_at - eye);
    right_ = Normalize(Cross(front_, up));
    up_ = Normalize(Cross(right_, front_));

    view_dx_ = right_ * tanf(ToRadians(0.5f * fov_x));
    view_dy_ = up_ * tanf(ToRadians(0.5f * fov_y_));

    spp_inv_ = 1.0f / spp;
}