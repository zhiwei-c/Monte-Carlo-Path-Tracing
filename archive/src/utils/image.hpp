#pragma once

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

Texture *LoadImage(const std::string &filename, const std::string &id = "", int *max_width = nullptr);

void SaveImage(const std::vector<float> &data, int width, int height, const std::string &filename);

inline double LinearRgbToLuminance(const dvec3 &rgb)
{
    return 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
}

NAMESPACE_END(raytracer)