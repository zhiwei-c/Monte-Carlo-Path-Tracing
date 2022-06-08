#include "texture.h"

__device__ int Modulo(int a, int b)
{
    auto c = a % b;
    if (c < 0)
        c += b;
    return c;
}

__device__ Float CyclicClamp(Float num)
{
    while (num > 1)
        num -= 1;
    while (num < 0)
        num += 1;
    return num;
}

__device__ void Texture::InitBitmap(int width, int height, int channel, float *colors)
{
    type_ = kBitmap;
    width_ = width;
    height_ = height;
    channel_ = channel;
    colors_ = colors;
}

__device__ void Texture::InitConstant(const vec3 &color)
{
    type_ = kConstant;
    color_ = color;
}

__device__ vec3 Texture::Color(const vec2 &texcoord) const
{
    switch (type_)
    {
    case kConstant:
        return color_;
        break;
    case kBitmap:
    {
        auto x = static_cast<int>(texcoord.x * width_),
             y = static_cast<int>(texcoord.y * height_);
        x = Modulo(x, width_);
        y = Modulo(y, height_);
        auto offset = (x + static_cast<uint>(width_) * y) * channel_;
        auto r = colors_[offset];
        auto g = colors_[offset + 1];
        auto b = colors_[offset + 2];
        return vec3(r, g, b);
        break;
    }
    default:
        break;
    }
    return vec3(0);
}

__device__ vec2 Texture::Gradient(const vec2 &texcoord) const
{
    switch (type_)
    {
    case kBitmap:
    {
        Float kh = 0.2, kn = 0.1;
        auto value = GetNorm(texcoord, 0, 0),
             value_u = GetNorm(texcoord, 1, 0),
             value_v = GetNorm(texcoord, 0, 1);
        auto du = kh * kn * (value_u - value),
             dv = kh * kn * (value_v - value);
        return vec2(du, dv);
        break;
    }
    default:
        break;
    }
    return vec2(0);
}

__device__ bool Texture::Transparent(const vec2 &texcoord, Float sample) const
{
    switch (type_)
    {
    case kConstant:
    {
        return sample > color_.x;
        break;
    }
    case kBitmap:
    {
        if (channel_ != 4)
            return false;
        auto x = static_cast<int>(texcoord.x * width_),
             y = static_cast<int>(texcoord.y * height_);
        x = Modulo(x, width_);
        y = Modulo(y, height_);
        auto offset = (x + width_ * y) * channel_;
        auto alpha = colors_[offset + 3];
        if (alpha == 1)
            return false;
        else if (alpha == 0)
            return true;
        else
            return sample > alpha;
        break;
    }
    default:
        break;
    }
    return false;
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
