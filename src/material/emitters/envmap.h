#pragma once

#include <memory>

#include "../textures/textures.h"

NAMESPACE_BEGIN(simple_renderer)

//环境光映射（实际上是“天空球”）
class Envmap
{
public:
    /**
     * \brief 恒定的环境光照
     * \param radiance 指定单位立体角单位面积的辐射亮度
     */
    Envmap(const Spectrum &radiance = Spectrum(0.3)) : to_local_(nullptr), radiance_(std::make_unique<ConstantTexture>(radiance)) {}

    /**
     * \brief 从图像加载捕获的环境光照信息，将其作为为无限远的光源。
     * \param file_path 待加载的辐射亮度输入图像的文件名
     * \param gamma 用于覆盖源位图的伽马值
     * \param to_world 从环境光照局部坐标系到世界坐标系的变换矩阵
     */
    Envmap(const std::string &file_path, Float gamma, std::unique_ptr<Mat4> to_world = nullptr)
        : to_local_(nullptr)
    {
        if (to_world)
            to_local_ = std::make_unique<Mat4>(glm::inverse(*to_world));

        if (!file_path.empty())
            radiance_.reset(new Bitmap(file_path, gamma));
        else
            radiance_.reset(new ConstantTexture(Spectrum(0.3)));
    }

    /**
     * \brief 给定方向的环境光辐射亮度
     * \param look_dir 观察方向，为环境光入射方向的反向
     * \return 获取到的环境光辐射亮度
     */
    Spectrum GetLe(Spectrum look_dir)
    {
        if (radiance_->Constant())
            return radiance_->GetPixel(Vector2(0));

        if (to_local_)
            look_dir = TransfromDir(*to_local_, look_dir);

        Float phi = 0, theta = 0;
        CartesianToSpherical(look_dir, theta, phi);

        phi = 2 * kPi - phi;
        while (phi > 2 * kPi)
            phi -= 2 * kPi;

        Vector2 coord;
        coord.x = static_cast<Float>(phi * 0.5 * kPiInv); // width
        coord.y = static_cast<Float>(theta * kPiInv);     // height

        return radiance_->GetPixel(coord);
    }

private:
    Float sampling_weight_;             //额外权重
    std::unique_ptr<Texture> radiance_; // 环境光辐射度纹理
    std::unique_ptr<Mat4> to_local_;    // 从世界坐标到环境光映射局部坐标的变换矩阵
};

NAMESPACE_END(simple_renderer)