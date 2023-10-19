#pragma once

#include "../tensor/tensor.cuh"

#include <vector>

struct Ray
{
    Vec3 origin;
    Vec3 dir;
    Vec3 dir_inv;

    QUALIFIER_DEVICE Ray(const Vec3 &origin, const Vec3 &dir)
        : origin(origin), dir(dir)
    {
        for (int i = 0; i < 3; ++i)
            dir_inv[i] = 1.0f / (dir[i] != 0 ? dir[i] : kEpsilonFloat);
    }
};

struct Hit
{
    bool valid;
    bool absorb;
    bool inside;
    float distance;
    float pdf_area;
    uint32_t id_instance;
    uint32_t id_bsdf;
    Vec2 texcoord;
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;

    QUALIFIER_DEVICE Hit()
        : valid(false), absorb(false), inside(false), distance(kMaxFloat), pdf_area(0),
          id_instance(kInvalidId), id_bsdf(kInvalidId), texcoord(Vec2()), position(Vec3()),
          normal(Vec3()), tangent(Vec3()), bitangent(Vec3())
    {
    }
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
