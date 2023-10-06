#include "sky.hpp"

#include "envmap.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../textures/bitmap.hpp"
#ifndef NDEBUG
#include "../utils/image.hpp"
#endif

extern "C"
{
#include <ArHosekSkyModel.h>
}

NAMESPACE_BEGIN(raytracer)

/* Apparent radius of the sun as seen from the earth (in degrees).
   This is an approximation--the actual value is somewhere between
   0.526 and 0.545 depending on the time of year */
constexpr double kSunAppRadius = 0.5358;

Sky::Sky(const dvec3 &sun_direction, const dvec3 &albedo, double turbidity, double stretch, double sun_scale,
         double sky_scale, double sun_radius_scale, int resolution, bool extend)
    : Emitter(EmitterType::kSky),
      envmap_(nullptr),
      background_(nullptr)
{
    int width = resolution;
    int height = resolution / 2;

    auto data = std::vector<float>(width * height * 3, 0);

    double zenith = std::acos(std::min(1.0, std::max(-1.0, sun_direction.y)));
    double azimuth = std::atan2(sun_direction.x, -sun_direction.z);
    if (azimuth < 0)
    {
        azimuth += 2.0 * kPi;
    }

    ArHosekSkyModelState *skymodel_state[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        skymodel_state[i] = arhosek_rgb_skymodelstate_alloc_init(
            turbidity,
            albedo[i],
            0.5f * kPi - zenith);
    }

    dvec2 factor = {2.0 * kPi / width, kPi / height};
    for (int y = 0; y < height; ++y)
    {
        double theta_raw = (y + 0.5) * factor.y;
        double theta = theta_raw / stretch;
        double factor2 = 1.0;
        if (std::cos(theta) <= 0.0)
        {
            if (extend)
            {
                theta = 0.5f * kPi - kEpsilonCompare;

                double temp1 = 2.0 - 2.0 * theta_raw * kPiRcp;
                temp1 = std::min(1.0, std::max(0.0, temp1));
                factor2 = temp1 * temp1 * (-2.0 * temp1 + 3);
            }
            else
            {
                continue;
            }
        }

        for (int x = 0; x < width; ++x)
        {
            double phi = (x + .5) * factor.x;
            double cos_gamma = std::cos(theta) * std::cos(zenith) + std::sin(theta) * std::sin(zenith) * std::cos(phi - azimuth);
            double gamma = std::acos(std::min(1.0, std::max(-1.0, cos_gamma)));

            dvec3 color;
            for (int i = 0; i < 3; ++i)
            {
                color[i] = std::max(0.0, arhosek_tristim_skymodel_radiance(skymodel_state[i],
                                                                           theta, gamma, i) /
                                             106.856980);
            }
            color *= sky_scale * factor2;
            const int offset = (x + y * width) * 3;
            data[offset] = static_cast<float>(color.r);
            data[offset + 1] = static_cast<float>(color.g);
            data[offset + 2] = static_cast<float>(color.b);
        }
    }

    for (int i = 0; i < 3; ++i)
    {
        arhosekskymodelstate_free(skymodel_state[i]);
    }

#ifndef NDEBUG
    SaveImage(data, width, height, "sunsky.png");
#endif

    background_ = new Bitmap("sunsky", data, width, height, 3);
    envmap_ = new Envmap(background_, 1, dmat4(1));
}

Sky::~Sky()
{
    if (envmap_ != nullptr)
    {
        delete envmap_;
        envmap_ = nullptr;
    }
    if (background_ != nullptr)
    {
        delete background_;
        background_ = nullptr;
    }
}

SamplingRecord Sky::Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                           Accelerator *accelerator) const
{
    return envmap_->Sample(its_shape, wo, sampler, accelerator);
}

double Sky::Pdf(const dvec3 &look_dir) const
{
    return envmap_->Pdf(look_dir);
}

dvec3 Sky::radiance(const dvec3 &position, const dvec3 &wi) const
{
    return envmap_->radiance(position, wi);
}

NAMESPACE_END(raytracer)