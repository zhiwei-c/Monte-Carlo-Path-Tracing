#pragma once

#include "../tensor/tensor.cuh"

#include <vector>

struct Ray
{
    float t_max;
    Vec3 origin;
    Vec3 dir;
    Vec3 dir_inv;

    QUALIFIER_DEVICE Ray(const Vec3 &origin, const Vec3 &dir);
};

QUALIFIER_DEVICE Vec3 Reflect(const Vec3 &wi, const Vec3 &normal);
QUALIFIER_DEVICE bool Refract(const Vec3 &wi, const Vec3 &normal, float eta_inv, Vec3 *wt);

QUALIFIER_DEVICE float FresnelSchlick(float cos_theta, float relectivity);
QUALIFIER_DEVICE Vec3 FresnelSchlick(float cos_theta, const Vec3 &relectivity);

QUALIFIER_DEVICE Vec3 AverageFresnelConductor(const Vec3 &reflectivity, const Vec3 &edgetint);
QUALIFIER_DEVICE float AverageFresnelDielectric(const float eta);

float EvalSpectrumAmplitude(const std::vector<float> &wavelengths,
                            const std::vector<float> &amplitudes, float lambda);
float AverageSpectrumSamples(const std::vector<float> &wavelengths,
                             const std::vector<float> &amplitudes,
                             float wavelength_start, float wavelength_end);
Vec3 SpectrumToRgb(const std::vector<float> &wavelengths, const std::vector<float> &amplitudes);
