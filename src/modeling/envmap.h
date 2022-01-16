#pragma once

#include <memory>

#include "../material/texture/bitmap.h"

NAMESPACE_BEGIN(simple_renderer)

//环境光映射（实际上是“天空球”）
class Envmap
{
public:
    /**
     * \brief 恒定的环境光照
     * \param radiance 指定单位立体角单位面积的辐射亮度
     */
    Envmap(const Spectrum &radiance = Spectrum(0.3)) : phi_offset_(0), to_local_(nullptr), texture_(nullptr), radiance_(radiance) {}

    /**
     * \brief 从图像加载捕获的环境光照信息，将其作为为无限远的光源。
     * \param file_path 待加载的辐射亮度输入图像的文件名
     * \param gamma 用于覆盖源位图的伽马值
     * \param phi_offset 环境光映射坐标在width方向的偏移
     */
    Envmap(const std::string &file_path, Float gamma, Float phi_offset = 0)
        : phi_offset_(phi_offset), to_local_(nullptr), texture_(nullptr), radiance_(Spectrum(0.3))
    {
        if (!file_path.empty())
            texture_.reset(new Bitmap(file_path, gamma));
    }

    /**
     * \brief 从图像加载捕获的环境光照信息，将其作为为无限远的光源。
     * \param file_path 待加载的辐射亮度输入图像的文件名
     * \param gamma 用于覆盖源位图的伽马值
     * \param to_world 从环境光照局部坐标系到世界坐标系的变换矩阵
     */
    Envmap(const std::string &file_path, Float gamma, std::unique_ptr<Mat4> to_world = nullptr)
        : phi_offset_(0), to_local_(nullptr), texture_(nullptr), radiance_(Spectrum(0.3))
    {
        if (to_world)
            to_local_ = std::make_unique<Mat4>(glm::inverse(*to_world));

        if (!file_path.empty())
            texture_.reset(new Bitmap(file_path, gamma));
    }

    /** 
     * \brief 给定方向的环境光辐射亮度
     * \param look_dir 观察方向，为环境光入射方向的反向
     * \return 获取到的环境光辐射亮度
     */
    Spectrum GetLe(Spectrum look_dir)
    {
        if (texture_ == nullptr)
            return radiance_;

        if (to_local_)
            look_dir = TransfromDir(*to_local_, look_dir);

        Float phi = 0, theta = 0;
        CartesianToSpherical(look_dir, theta, phi);

        phi = 2 * kPi - phi + phi_offset_;
        while (phi > 2 * kPi)
            phi -= 2 * kPi;

        Vector2 coord;
        coord.x = static_cast<Float>(phi * 0.5 * kPiInv); // width
        coord.y = static_cast<Float>(theta * kPiInv);     // height

        return texture_->GetPixel(coord);
    }

private:
    Spectrum radiance_;                // 恒定的环境光辐射度
    std::unique_ptr<Bitmap> texture_; // 环境光辐射度纹理
    Float phi_offset_;                // 环境光映射坐标在width方向的偏移
    std::unique_ptr<Mat4> to_local_;  // 从世界坐标到环境光映射局部坐标的变换矩阵
};

NAMESPACE_END(simple_renderer)