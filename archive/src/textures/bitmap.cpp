#include "bitmap.hpp"

#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

Bitmap::Bitmap(const std::string &id, const std::vector<float> &data, int width, int height, int channels)
    : Texture(TextureType::kBitmap, id),
      data_(data),
      width_(width),
      height_(height),
      channels_(channels)
{
}

dvec3 Bitmap::color(const dvec2 &texcoord) const
{
    double x = texcoord.x * width_, y = texcoord.y * height_;
    while (x < 0)
        x += width_;
    while (x > width_ - 1)
        x -= width_;
    while (y < 0)
        y += height_;
    while (y > height_ - 1)
        y -= height_;

    int x_lower = static_cast<int>(x),
        y_lower = static_cast<int>(y);

    double t_x = x - x_lower, t_y = y - y_lower;

    int x_upper = x_lower, y_upper = y_lower;
    if (t_x > 0.0)
        x_upper += 1;
    if (t_y > 0.0)
        y_upper += 1;

    int offset = (x_lower + width_ * y_lower) * channels_;
    dvec3 color_lower_lower = (channels_ == 1) ? dvec3(data_[offset]) : dvec3{data_[offset], data_[offset + 1], data_[offset + 2]};

    offset = (x_lower + width_ * y_upper) * channels_;
    dvec3 color_lower_upper = (channels_ == 1) ? dvec3(data_[offset]) : dvec3{data_[offset], data_[offset + 1], data_[offset + 2]};

    offset = (x_upper + width_ * y_lower) * channels_;
    dvec3 color_upper_lower = (channels_ == 1) ? dvec3(data_[offset]) : dvec3{data_[offset], data_[offset + 1], data_[offset + 2]};

    offset = (x_upper + width_ * y_upper) * channels_;
    dvec3 color_upper_upper = (channels_ == 1) ? dvec3(data_[offset]) : dvec3{data_[offset], data_[offset + 1], data_[offset + 2]};

    dvec3 color_lower_lerp = Lerp(t_y, color_lower_lower, color_lower_upper);
    dvec3 color_upper_lerp = Lerp(t_y, color_upper_lower, color_upper_upper);

    const dvec3 color_lerp = Lerp(t_x, color_lower_lerp, color_upper_lerp);
    
    return color_lerp;
}

dvec2 Bitmap::gradient(const dvec2 &texcoord) const
{
    auto GetNorm = [&](const dvec2 &texcoord, int offset_x, int offset_y) -> float
    {
        const int x = Modulo(static_cast<int>(texcoord.x * width_) + offset_x, width_),
                  y = Modulo(static_cast<int>(texcoord.y * height_) + offset_y, height_),
                  offset = (x + width_ * y) * channels_;
        if (channels_ == 1)
        {
            return 255 * data_[offset];
        }
        else
        {
            float r = data_[offset],
                  g = data_[offset + 1],
                  b = data_[offset + 2];
            return std::sqrt(r * r + g * g + b * b);
        }
    };
    float kh = 2.1f, kn = 2.1f,
          value = GetNorm(texcoord, 0, 0),
          value_u = GetNorm(texcoord, 1, 0),
          value_v = GetNorm(texcoord, 0, 1);
    float du = kh * kn * (value_u - value),
          dv = kh * kn * (value_v - value);
    return {du, dv};
}

bool Bitmap::IsTransparent(const dvec2 &texcoord, Sampler* sampler) const
{
    if (channels_ != 4)
    {
        return false;
    }
    const int x = Modulo(static_cast<int>(texcoord.x * width_), width_),
              y = Modulo(static_cast<int>(texcoord.y * height_), height_),
              offset = (x + width_ * y) * channels_;
    const float alpha = data_[offset + 3];
    return sampler->Next1D() >= alpha;
}

NAMESPACE_END(raytracer)