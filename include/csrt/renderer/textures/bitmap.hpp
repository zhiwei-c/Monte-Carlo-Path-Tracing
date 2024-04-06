#ifndef CSRT__RENDERER__TEXTURES__BITMAP_HPP
#define CSRT__RENDERER__TEXTURES__BITMAP_HPP

#include <vector>

#include "../../tensor.hpp"
#include "../../utils.hpp"

namespace csrt
{

struct BitmapInfo
{
    int width = 0;
    int height = 0;
    int channel = 0;
    std::vector<float> data = {};
    Mat4 to_uv = {};
};

struct BitmapData
{
    int width = 0;
    int height = 0;
    int channel = 0;
    float *data = nullptr;
    Mat4 to_uv = {};
};

QUALIFIER_D_H Vec3 GetColorBitmap(const BitmapData &data, const Vec2 &texcoord);

QUALIFIER_D_H Vec2 GetGradientBitmap(const BitmapData &data,
                                     const Vec2 &texcoord);

QUALIFIER_D_H bool IsTransparentBitmap(const BitmapData &data,
                                       const Vec2 &texcoord, uint32_t *seed);

} // namespace csrt

#endif