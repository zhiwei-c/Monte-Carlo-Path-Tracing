#ifndef CSRT__PARSER__SUN_KSY_HPP
#define CSRT__PARSER__SUN_KSY_HPP

#include <vector>

#include "../tensor.hpp"
#include "spectrum.hpp"

namespace csrt
{

/* Apparent radius of the sun as seen from the earth (in degrees).
   This is an approximation--the actual value is somewhere between 0.526 and 0.545 depending on the time of year */
constexpr float kSunAppRadius = 0.5358;

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
float *CreateSunTexture(const Vec3 &sun_direction, const float turbidity,
                        const float radiance_scale, const float radius_scale,
                        const int width, const int height, Vec3 *radiance);
float *CreateSkyTexture(const Vec3 &sun_direction, const Vec3 &albedo,
                        const float turbidity, const float stretch,
                        const float radiance_scale, const bool extend,
                        const int width, const int height);

} // namespace csrt

#endif