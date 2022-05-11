#pragma once

#include <string>

#include "../core/texture_base.h"

NAMESPACE_BEGIN(raytracer)

//位图派生类
class Bitmap : public Texture
{

public:
    ///\brief 空白位图
    ///\param width - 图像宽度
    ///\param height - 图像高度
    ///\param channels - 图像的通道数
    Bitmap(int width, int height, int channels, Float gamma);

    ///\brief 位图
    ///\param filename - 加载图像文件名
    ///\param gamma - 伽马值
    Bitmap(const std::string &filename, Float gamma);

    ///\return 纹理在给定坐标处像素值
    Spectrum Color(const Vector2 &coord) const override;

    ///\return 纹理在给定坐标处梯度
    Vector2 Gradient(const Vector2 &coord) const override;

    ///\return 材质在给定的纹理坐标处是否透明
    bool Transparent(const Vector2 &coord) const override;

    ///\brief 设置纹理在给定坐标处像素值
    void SetColor(int x, int y, const Vector3 &value);

    ///\brief 保存图像到指定路径
    void Save(const std::string &path);

private:
    int width_;               //图像的宽
    int height_;              //图像的高
    int channels_;            //图像的通道数
    Float gamma_;             //对颜色值进行非线性映射的系数
    Float gamma_inv_;         //对颜色值进行非线性映射的系数的导数
    std::vector<Float> data_; //图像的数据
};

NAMESPACE_END(raytracer)