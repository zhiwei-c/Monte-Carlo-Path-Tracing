#pragma once

#include <vector>
#include <string>

#include "global.h"

inline static Float UndoGamma(Float value, Float gamma)
{
    if (gamma == -1)
    {
        if (value <= (Float)0.04045)
            return value * (Float)(1.0 / 12.92);
        else
            return std::pow((Float)((value + (Float)0.055) * (Float)(1.0 / 1.055)), (Float)2.4);
    }
    else
        return std::pow(value, gamma);
}

__device__ inline Float ApplyGamma(Float value, Float gamma_inv)
{
    if (gamma_inv == -1)
        return (value <= (Float)0.0031308) ? ((Float)12.92 * value)
                                           : ((Float)1.055 * pow(value, (Float)(1.0 / 2.4)) - (Float)0.055);
    else
        return pow(value, gamma_inv);
}

struct Frame
{
    int width;
    int height;
    std::vector<float> data;

    Frame() : width(0), height(0) {}
};

void ImageResize(int old_width, int old_height, int new_width, int new_height, int channels, std::vector<float> &colors);

void ImageReader(const std::string &filename, Float gamma, int &width, int &height, int &channels, std::vector<float> &colors);

int WriteOpenexr(Frame &frame, const std::string &path);

void WriteImage(Frame &frame, const std::string &path);