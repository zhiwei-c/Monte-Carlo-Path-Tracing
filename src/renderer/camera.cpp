#include "csrt/renderer/camera.hpp"

#include <cmath>

#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H Camera::Camera()
    : spp_(64), width_(1024), height_(1024), fov_x_(19.5f),
      eye_{0.0f, 1.0f, 6.8f}, up_{0.0f, 1.0f, 0.0f}
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

QUALIFIER_D_H Camera::Camera(const Camera::Info &info)
    : spp_(info.spp), spp_inv_(1.0f / info.spp), width_(info.width),
      height_(info.height), fov_x_(info.fov_x), eye_(info.eye), up_(info.up)
{
    fov_y_ = info.fov_x * info.height / info.width;
    front_ = Normalize(info.look_at - info.eye);
    right_ = Normalize(Cross(front_, info.up));
    up_ = Normalize(Cross(right_, front_));

    view_dx_ = right_ * tanf(ToRadians(0.5f * info.fov_x));
    view_dy_ = up_ * tanf(ToRadians(0.5f * fov_y_));
}

} // namespace csrt