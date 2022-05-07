#pragma once

#include "../global.h"

__device__ inline Float MisWeight(Float pdf1, Float pdf2)
{
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return pdf1 / (pdf1 + pdf2);
}

__device__ inline void HemisCos(Float x_1, Float x_2, vec3 &dir, Float &pdf)
{
    auto cos_theta = sqrt(x_1),
          phi = 2.0 * kPi * x_2;
    auto sin_theta = sqrt(1.0 - cos_theta * cos_theta),
          cos_phi = cos(phi),
          sin_phi = sin(phi);
    dir = vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    pdf = kPiInv * cos_theta;
}

__device__ inline Float PdfHemisCos(const vec3 &dir_local)
{
    auto cos_theta = dir_local.z;
    auto pdf = kPiInv * cos_theta;
    return pdf;
}
