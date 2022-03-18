#pragma once

#include <string>

#include "../texture.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

class Bitmap : public Texture
{

public:
    // \brief 加载图像
    //
    // \param file_name - 图像文件名
    Bitmap(const std::string &file_name, Float gamma);

    // \brief 初始化空白图像
    //
    // \param width - 图像宽度
    //
    // \param height - 图像高度
    //
    // \param channels - 图像的通道数
    Bitmap(int width, int height, int channels, Float gamma);

    // \brief 根据坐标更新图像在相应位置的数值
    //
    // \param x - width 方向的坐标，范围[0,width-1]
    //
    // \param y - height 方向的坐标，范围[0,height-1]
    //
    // \param value 图像在相应位置新的数值
    void SetPixel(int x, int y, const Vector3 &value);

    // \brief 根据坐标获取图像在相应位置的数值
    //
    // \param pos_x - width 方向的坐标，范围[0,1]
    //
    // \param pos_y - height 方向的坐标，范围[0,1]
    //
    // \return 图像在相应位置的数值
    Spectrum GetPixel(const Vector2 &coord) const override;

    //\param pos 给定位置图像坐标
    //
    //\return 图像在给定位置处是否完全透明
    bool Transparent(const Vector2 &coord) const override;

    Vector2 GetGradient(const Vector2 &coord) const override;

    // \brief 保存数据为 png 格式图像
    //
    // \param file_name - 图像文件名
    void Write(const std::string &path);

private:
    int width_;       //图像的宽
    int height_;      //图像的高
    int channels_;    //图像的通道数
    Float gamma_;     //对颜色值进行非线性映射的系数
    Float gamma_inv_; //对颜色值进行非线性映射的系数的导数
    std::string file_name_;
    std::vector<Float> data_; //图像的数据

    int WriteOpenexr(const std::string &path);
};

NAMESPACE_END(simple_renderer)