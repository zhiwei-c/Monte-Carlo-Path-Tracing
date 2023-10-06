#pragma once

#include "../tensor/tensor.cuh"
#include "directional_light.cuh"

struct LocationDate
{
    int year = 2010;
    int month = 7;
    int day = 10;
    float hour = 15.0f;
    float minute = 0.0f;
    float second = 0.0f;
    float timezone = 9.0f;
    float latitude = 35.6894f;
    float longitude = 139.6917f;
};

Vec3 GetSunDirection(const LocationDate &location_date);
void CreateSunTexture(const Vec3 &sun_direction, float turbidity, float sun_scale,
                      float sun_radius_scale, int width, int height, Vec3 *radiance,
                      std::vector<float> *data);
void CreateSkyTexture(const Vec3 &sun_direction, const Vec3 &albedo, float turbidity,
                      float stretch, float sky_scale, bool extend, int width, int height,
                      std::vector<float> *data);
