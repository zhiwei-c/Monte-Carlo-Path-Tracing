#pragma once

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

struct LocationDataInfo
{
    int year;
    int month;
    int day;
    double hour;
    double minute;
    double second;
    double timezone;
    double latitude;
    double longitude;

    LocationDataInfo();
};

dvec3 GetSunDirection(const LocationDataInfo &location_time);

dvec3 GetSunRadiance(double theta, double turbidity);

NAMESPACE_END(raytracer)