#include "textures/textures.h"

__device__ vec3 Texture::Color(const vec2 &texcoord) const
{
    switch (type_)
    {
    case kConstant:
        return color_;
        break;
    case kBitmap:
        return ColorBitmap(texcoord);
        break;
    case kCheckerboard:
        return ColorCheckerboard(texcoord);
        break;
    case kGridTexture:
        return ColorGridTexture(texcoord);
        break;
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
        return GradientBitmap(texcoord);
        break;
    case kCheckerboard:
        return GradientCheckerboard(texcoord);
        break;
    case kGridTexture:
        return GradientGridTexture(texcoord);
        break;
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
