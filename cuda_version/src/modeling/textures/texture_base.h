#pragma once

#include "texture_info.h"

class Texture
{
public:
    __device__ Texture()
        : type_(kNoneTexture),
          width_(0),
          height_(0),
          channel_(0),
          color_(vec3(0)),
          colors_(nullptr),
          color0_(vec3(0.4)),
          color1_(vec3(0.2)),
          uv_scale_(nullptr),
          uv_offset_(nullptr),
          line_width_(0) {}

    __device__ ~Texture()
    {
        if (uv_scale_)
        {
            delete uv_scale_;
            uv_scale_ = nullptr;
        }
        if (uv_offset_)
        {
            delete uv_offset_;
            uv_offset_ = nullptr;
        }
    }

    __device__ vec3 Color(const vec2 &texcoord) const;

    __device__ vec2 Gradient(const vec2 &texcoord) const;

    __device__ bool Transparent(const vec2 &texcoord, Float sample) const;

    __device__ TextureType type() const { return type_; }

    __device__ void InitConstant(const vec3 &color);

    __device__ void InitBitmap(int width, int height, int channel, float *colors);

    __device__ void InitCheckerboard(const vec3 &color0, const vec3 &color1, vec2 *uv_scale, vec2 *uv_offset);

    __device__ void InitGridTexture(const vec3 &color0, const vec3 &color1, Float line_width, vec2 *uv_scale, vec2 *uv_offset);

private:
    TextureType type_;
    int width_;
    int height_;
    int channel_;
    vec3 color_;
    vec3 color0_;
    vec3 color1_;
    vec2 *uv_scale_;
    vec2 *uv_offset_;
    Float line_width_;
    float *colors_;

    __device__ Float GetNorm(const vec2 &coord, int offset_x, int offset_y) const;

    __device__ vec3 ColorBitmap(const vec2 &texcoord) const;
    __device__ vec2 GradientBitmap(const vec2 &texcoord) const;

    __device__ vec3 ColorCheckerboard(const vec2 &texcoord) const;
    __device__ vec2 GradientCheckerboard(const vec2 &texcoord) const;

    __device__ vec3 ColorGridTexture(const vec2 &texcoord) const;
    __device__ vec2 GradientGridTexture(const vec2 &texcoord) const;
};

__device__ inline int Modulo(int a, int b)
{
    auto c = a % b;
    if (c < 0)
        c += b;
    return c;
}

__device__ inline Float CyclicClamp(Float num)
{
    while (num > 1)
        num -= 1;
    while (num < 0)
        num += 1;
    return num;
}

__device__ Float Texture::GetNorm(const vec2 &coord, int offset_x, int offset_y) const
{
    auto x = static_cast<int>(coord.x * width_) + offset_x,
         y = static_cast<int>(coord.y * height_) + offset_y;
    x = Modulo(x, width_);
    y = Modulo(y, height_);
    auto offset = (x + width_ * y) * channel_;
    auto r = 255 * colors_[offset];
    auto g = 255 * colors_[offset + 1];
    auto b = 255 * colors_[offset + 2];
    return myvec::length(vec3(r, g, b));
};
