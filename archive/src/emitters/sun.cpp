#include "sun.hpp"

#include <tuple>
#include <unordered_map>

#include "directional_emitter.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../textures/bitmap.hpp"
#include "../utils/image.hpp"
#include "../utils/sun_model.hpp"

NAMESPACE_BEGIN(raytracer)

/* Apparent radius of the sun as seen from the earth (in degrees).
   This is an approximation--the actual value is somewhere between
   0.526 and 0.545 depending on the time of year */
constexpr double kSunAppRadius = 0.5358;

Sun::Sun(const dvec3 &sun_direction, double turbidity, int resolution, double sun_scale, double sun_radius_scale)
    : Emitter(EmitterType::kSun),
      resolution_(resolution)
{
    double zenith = std::acos(std::min(1.0, std::max(-1.0, sun_direction.y)));
    double azimuth = std::atan2(sun_direction.x, -sun_direction.z);
    if (azimuth < 0)
    {
        azimuth += 2.0 * kPi;
    }

    double theta = glm::radians(kSunAppRadius * 0.5);

    dvec3 sun_radiance = GetSunRadiance(zenith, turbidity) * sun_scale;
    double solid_angle = 2.0 * kPi * (1 - std::cos(theta));
    if (sun_radius_scale <= 0.0)
    {
        dvec3 radiance = sun_radiance * solid_angle;
        directionals_.push_back(new DistantDirectionalEmitter(radiance, -sun_direction));
        cdf_ = {0, 1};
        normalization_ = 1.0 / LinearRgbToLuminance(radiance);
    }
    else
    {
        double cos_theta = std::cos(theta * sun_radius_scale);
        double covered_portion = 0.5 * (1 - cos_theta); /* Ratio of the sphere that is covered by the sun */
        int width = resolution, height = resolution / 2;
        auto pixel_count = static_cast<size_t>(width * height * 0.5);

        auto sample_num = static_cast<int>(std::max(100.0, (pixel_count * covered_portion * 1000)));

        dvec3 value = sun_radiance * solid_angle / static_cast<double>(sample_num);
        auto factor = dvec2{width * 0.5 * kPiRcp, height * kPiRcp};

        std::unordered_map<int, dvec3> mp;
        for (int i = 0; i < sample_num; ++i)
        {
            dvec3 dir = SampleConeUniform(cos_theta, Hammersley(i + 1, sample_num + 1));
            dir = ToWorld(dir, sun_direction);

            double local_azimuth = std::atan2(dir.x, -dir.z),
                   local_elevation = std::acos(std::min(1.0, std::max(-1.0, dir.y)));
            if (local_azimuth < 0)
            {
                local_azimuth += 2.0 * kPi;
            }
            const int x = std::min(std::max(0, (int)(local_azimuth * factor.x)), width - 1),
                      y = std::min(std::max(0, (int)(local_elevation * factor.y)), height - 1);
            const int offset = x + y * width;
            mp[offset] += value;
        }

        cdf_ = std::vector<double>(mp.size() + 1, 0);
        sun_pixels_.reserve(mp.size());
        pixel_colors_.reserve(mp.size());
        size_t cdf_index = 1;
        for (auto [offset, color] : mp)
        {
            int x = offset % width, y = offset / width;
            sun_pixels_.push_back(std::pair{x, y});
            pixel_colors_.push_back(color);

            double phi = x * 2.0 * kPi / width,
                   theta = y * kPi / height;
            auto look_dir = dvec3{std::sin(phi) * std::sin(theta), std::cos(theta), -std::cos(phi) * std::sin(theta)};
            directionals_.push_back(new DistantDirectionalEmitter(color, -look_dir));
            cdf_[cdf_index] = cdf_[cdf_index - 1] + LinearRgbToLuminance(color);
            ++cdf_index;
        }

        normalization_ = 1.0 / cdf_.back();
        for (size_t i = 1; i < cdf_.size() - 1; ++i)
        {
            cdf_[i] *= normalization_;
        }
        cdf_.back() = 1.0;
    }
}

Sun::~Sun()
{
    for (auto &directional : directionals_)
    {
        delete directional;
        directional = nullptr;
    }
}

SamplingRecord Sun::Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                           Accelerator *accelerator) const
{
    size_t index = SampleCdf(cdf_, directionals_.size(), sampler->Next1D());
    SamplingRecord rec = directionals_[index]->Sample(its_shape, wo, sampler, accelerator);
    if (rec.type != ScatteringType::kNone)
    {
        rec.radiance /= LinearRgbToLuminance(rec.radiance) * normalization_;
    }
    return rec;
}
dvec3 Sun::radiance(const dvec3 &position, const dvec3 &wi) const
{
    const int width = resolution_, height = resolution_ / 2;
    auto factor = dvec2{width * 0.5 * kPiRcp, height * kPiRcp};
    dvec3 dir = -wi;
    double azimuth = std::atan2(dir.x, -dir.z),
           elevation = std::acos(std::min(1.0, std::max(-1.0, dir.y)));
    if (azimuth < 0)
    {
        azimuth += 2.0 * kPi;
    }
    const int x = std::min(std::max(0, (int)(azimuth * factor.x)), width - 1),
              y = std::min(std::max(0, (int)(elevation * factor.y)), height - 1);
    for (size_t i = 0; i < sun_pixels_.size(); ++i)
    {
        if (sun_pixels_[i].first == x && sun_pixels_[i].second == y)
        {
            return pixel_colors_[i];
        }
    }
    return dvec3(0);
}

NAMESPACE_END(raytracer)