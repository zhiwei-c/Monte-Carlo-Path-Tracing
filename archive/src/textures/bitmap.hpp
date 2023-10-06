#pragma once

#include <vector>

#include "texture.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//纹理派生类，位图
class Bitmap : public Texture
{
public:
    Bitmap(const std::string &id, const std::vector<float> &data, int width, int height, int channels);

    dvec3 color(const dvec2 &texcoord) const override;
    dvec2 gradient(const dvec2 &texcoord) const override;
    bool IsTransparent(const dvec2 &texcoord, Sampler* sampler) const override;
    int width() const override { return width_; }
    int height() const override { return height_; }

private:
    int width_;               //宽度
    int height_;              //高度
    int channels_;            //通道数
    std::vector<float> data_; //数据
};

NAMESPACE_END(raytracer)