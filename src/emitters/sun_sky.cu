#include "sun_sky.cuh"

#include <unordered_map>
#include <utility>

#include "../utils/math.cuh"
#include "../renderer/ray.cuh"

extern "C"
{
#include <ArHosekSkyModel.h>
}

namespace
{
    /* Apparent radius of the sun as seen from the earth (in degrees).
       This is an approximation--the actual value is somewhere between
       0.526 and 0.545 depending on the time of year */
    constexpr float kSunAppRadius = 0.5358;

    constexpr float kEarthMeanRadius = 6371.01f;   // In km
    constexpr float kAstronomicalUnit = 149597890; // In km

    /* The following is from the implementation of "A Practical Analytic Model for
       Daylight" by A.J. Preetham, Peter Shirley, and Brian Smits */

    /* All data lifted from MI. Units are either [] or cm^-1. refer when in doubt MI */

    // k_o Spectrum table from pg 127, MI.
    constexpr float kKoWavelengths[64] = {
        300, 305, 310, 315, 320, 325, 330, 335, 340, 345,
        350, 355, 445, 450, 455, 460, 465, 470, 475, 480,
        485, 490, 495, 500, 505, 510, 515, 520, 525, 530,
        535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
        585, 590, 595, 600, 605, 610, 620, 630, 640, 650,
        660, 670, 680, 690, 700, 710, 720, 730, 740, 750,
        760, 770, 780, 790};

    constexpr float kKoAmplitudes[64] = {
        10.0, 4.8, 2.7, 1.35, .8, .380, .160, .075, .04, .019, .007,
        .0, .003, .003, .004, .006, .008, .009, .012, .014, .017,
        .021, .025, .03, .035, .04, .045, .048, .057, .063, .07,
        .075, .08, .085, .095, .103, .110, .12, .122, .12, .118,
        .115, .12, .125, .130, .12, .105, .09, .079, .067, .057,
        .048, .036, .028, .023, .018, .014, .011, .010, .009,
        .007, .004, .0, .0};

    // k_g Spectrum table from pg 130, MI.
    constexpr float kKgWavelengths[4] = {
        759, 760, 770, 771};

    constexpr float kKgAmplitudes[4] = {
        0, 3.0, 0.210, 0};

    // k_wa Spectrum table from pg 130, MI.
    constexpr float kKwaWavelengths[13] = {
        689, 690, 700, 710, 720,
        730, 740, 750, 760, 770,
        780, 790, 800};

    constexpr float kKwaAmplitudes[13] = {
        0, 0.160e-1, 0.240e-1, 0.125e-1,
        0.100e+1, 0.870, 0.610e-1, 0.100e-2,
        0.100e-4, 0.100e-4, 0.600e-3,
        0.175e-1, 0.360e-1};

    /* Wavelengths corresponding to the table below */
    constexpr float kSolarWavelengths[38] = {
        380, 390, 400, 410, 420, 430, 440, 450,
        460, 470, 480, 490, 500, 510, 520, 530,
        540, 550, 560, 570, 580, 590, 600, 610,
        620, 630, 640, 650, 660, 670, 680, 690,
        700, 710, 720, 730, 740, 750};

    /* Solar amplitude in watts / (m^2 * nm * sr) */
    constexpr float kSolarAmplitudes[38] = {
        16559.0, 16233.7, 21127.5, 25888.2, 25829.1,
        24232.3, 26760.5, 29658.3, 30545.4, 30057.5,
        30663.7, 28830.4, 28712.1, 27825.0, 27100.6,
        27233.6, 26361.3, 25503.8, 25060.2, 25311.6,
        25355.9, 25134.2, 24631.5, 24173.2, 23685.3,
        23212.1, 22827.7, 22339.8, 21970.2, 21526.7,
        21097.9, 20728.3, 20240.4, 19870.8, 19427.2,
        19072.4, 18628.9, 18259.2};

    Vec3 GetSunRadiance(float theta, float turbidity)
    {
        static std::vector<float>
            k_o_wavelengths = {kKoWavelengths,
                               kKoWavelengths + sizeof(kKoWavelengths) / sizeof(kKoWavelengths[0])},
            k_o_amplitudes = {kKoAmplitudes,
                              kKoAmplitudes + sizeof(kKoAmplitudes) / sizeof(kKoAmplitudes[0])},
            k_g_wavelengths = {kKgWavelengths,
                               kKgWavelengths + sizeof(kKgWavelengths) / sizeof(kKgWavelengths[0])},
            K_g_values = {kKgAmplitudes,
                          kKgAmplitudes + sizeof(kKgAmplitudes) / sizeof(kKgAmplitudes[0])},
            wa_wavelengths = {kKwaWavelengths,
                              kKwaWavelengths + sizeof(kKwaWavelengths) / sizeof(kKwaWavelengths[0])},
            wa_amplitudes = {kKwaAmplitudes,
                             kKwaAmplitudes + sizeof(kKwaAmplitudes) / sizeof(kKwaAmplitudes[0])},
            solar_wavelengths = {kSolarWavelengths,
                                 kSolarWavelengths + sizeof(kSolarWavelengths) /
                                                         sizeof(kSolarWavelengths[0])},
            solar_amplitudes = {kSolarAmplitudes,
                                kSolarAmplitudes + sizeof(kSolarAmplitudes) /
                                                       sizeof(kSolarAmplitudes[0])};

        std::vector<float> data(91), wavelengths(91); // (800 - 350) / 5  + 1 = 91
        float beta = 0.04608365822050f * turbidity - 0.04586025928522f;

        // Relative Optical Mass
        float m = 1.0f / (std::cos(theta) + 0.15f * std::pow(93.885 - theta / kPi * 180.0, -1.253f));

        float lambda;
        int i = 0;
        for (i = 0, lambda = 350; i < 91; ++i, lambda += 5)
        {
            // Rayleigh Scattering
            // Results agree with the graph (pg 115, MI) */
            float tau_rR = std::exp(-m * 0.008735f * std::pow(lambda / 1000.0f, -4.08f));

            // Aerosol (water + dust) attenuation
            // beta - amount of aerosols present
            // alpha - ratio of small to large particle sizes. (0:4,usually 1.3)
            // Results agree with the graph (pg 121, MI)
            const float alpha = 1.3f;
            // lambda should be in um
            float tau_a = std::exp(-m * beta * std::pow(lambda / 1000.0, -alpha));

            // Attenuation due to ozone absorption
            // l_ozone - amount of ozone in cm(NTP)
            // Results agree with the graph (pg 128, MI)
            const float l_ozone = 0.35;
            const float k_o_amplitude = EvalSpectrumAmplitude(k_o_wavelengths, k_o_amplitudes, lambda);
            float tau_o = std::exp(-m * k_o_amplitude * l_ozone);

            // Attenuation due to mixed gases absorption
            // Results agree with the graph (pg 131, MI)
            const float k_g_amplitude = EvalSpectrumAmplitude(k_g_wavelengths, K_g_values, lambda);
            float tau_g = std::exp(-1.41f * k_g_amplitude * m /
                                   std::pow(1 + 118.93f * k_g_amplitude * m, 0.45));

            // Attenuation due to water vapor absorbtion
            // w - precipitable water vapor in centimeters (standard = 2)
            // Results agree with the graph (pg 132, MI)
            const float w = 2.0;
            const float wa_amplitude = EvalSpectrumAmplitude(wa_wavelengths, wa_amplitudes, lambda);
            float tau_va = std::exp(-0.2385 * wa_amplitude * w * m /
                                    std::pow(1 + 20.07 * wa_amplitude * w * m, 0.45));

            wavelengths[i] = lambda;
            data[i] = EvalSpectrumAmplitude(solar_wavelengths, solar_amplitudes, lambda) *
                      tau_rR * tau_a * tau_o * tau_g * tau_va;
        }

        Vec3 rgb = SpectrumToRgb(wavelengths, data);
        return rgb;
    }

} // namespace

Vec3 GetSunDirection(const LocationDate &location_date)
{
    // Main variables
    float elapsed_Julian_days, dec_hours;
    float ecliptic_longitude, ecliptic_obliquity;
    float right_scension, declination;
    float zenith, azimuth;

    // Auxiliary variables
    float dx, dy;

    /* Calculate difference in days between the current Julian Day
       and JD 2451545.0, which is noon 1 January 2000 Universal Time */
    {
        // Calculate time of the day in UT decimal hours
        dec_hours = location_date.hour - location_date.timezone +
                    (location_date.minute + location_date.second / 60.0f) / 60.0f;

        // Calculate current Julian Day
        int li_aux1 = (location_date.month - 14) / 12;
        int li_aux2 = (1461 * (location_date.year + 4800 + li_aux1)) / 4 +
                      (367 * (location_date.month - 2 - 12 * li_aux1)) / 12 -
                      (3 * ((location_date.year + 4900 + li_aux1) / 100)) / 4 +
                      location_date.day - 32075;
        float d_Julian_date = li_aux2 - 0.5f + dec_hours / 24.0f;

        // Calculate difference between current Julian Day and JD 2451545.0
        elapsed_Julian_days = d_Julian_date - 2451545.0;
    }

    /* Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
       ecliptic in radians but without limiting the angle to be less than 2*Pi
       (i.e., the result may be greater than 2*Pi) */
    {
        float omega = 2.1429f - 0.0010394594f * elapsed_Julian_days;
        float mean_longitude = 4.8950630f + 0.017202791698f * elapsed_Julian_days; // Radians
        float anomaly = 6.2400600f + 0.0172019699f * elapsed_Julian_days;

        ecliptic_longitude = mean_longitude + 0.03341607f * sinf(anomaly) +
                             0.00034894f * sinf(2 * anomaly) - 0.0001134f - 0.0000203f * sinf(omega);

        ecliptic_obliquity = 0.4090928f - 6.2140e-9f * elapsed_Julian_days + 0.0000396f * cosf(omega);
    }

    /* Calculate celestial coordinates ( right ascension and declination ) in radians
       but without limiting the angle to be less than 2*Pi (i.e., the result may be
       greater than 2*Pi) */
    {
        float sin_eclipticLongitude = sinf(ecliptic_longitude);
        dy = cosf(ecliptic_obliquity) * sin_eclipticLongitude;
        dx = cosf(ecliptic_longitude);
        right_scension = atan2f(dy, dx);
        if (right_scension < 0.0f)
            right_scension += 2.0f * kPi;

        declination = asinf(sinf(ecliptic_obliquity) * sin_eclipticLongitude);
    }

    // Calculate local coordinates (azimuth and zenith angle) in degrees
    {
        float greenwich_Mean_sidereal_time = 6.6974243242f + 0.0657098283f * elapsed_Julian_days +
                                             dec_hours;

        float local_mean_sidereal_time = ToRadians(greenwich_Mean_sidereal_time * 15.0f +
                                                   location_date.longitude);

        float latitude_in_radians = ToRadians(location_date.latitude);
        float cos_latitude = cosf(latitude_in_radians);
        float sin_latitude = sinf(latitude_in_radians);

        float hour_angle = local_mean_sidereal_time - right_scension;
        float cos_hour_angle = cosf(hour_angle);

        zenith = acosf(cos_latitude * cos_hour_angle * cosf(declination) +
                       sinf(declination) * sin_latitude);

        dy = -sinf(hour_angle);
        dx = tanf(declination) * cos_latitude - sin_latitude * cos_hour_angle;

        azimuth = atan2f(dy, dx);
        if (azimuth < 0.0f)
            azimuth += 2.0f * kPi;

        // Parallax Correction
        zenith += (kEarthMeanRadius / kAstronomicalUnit) * sinf(zenith);
    }

    return {sinf(azimuth) * sinf(zenith), cosf(zenith), -cosf(azimuth) * sinf(zenith)};
}

void CreateSunTexture(const Vec3 &sun_direction, float turbidity, float sun_scale,
                      float sun_radius_scale, int width, int height, Vec3 *radiance,
                      std::vector<float> *data)
{
    const float theta = ToRadians(kSunAppRadius * 0.5f),
                solid_angle = 2.0f * kPi * (1.0f - cosf(theta)),
                zenith = acosf(fminf(1.0f, fmaxf(-1.0f, sun_direction.y)));
    const Vec3 sun_radiance = ::GetSunRadiance(zenith, turbidity) * sun_scale;
    *radiance = sun_radiance * solid_angle;

    const float cos_theta = cosf(theta * sun_radius_scale),
                covered_portion = 0.5f * (1.0f - cos_theta); // sphere ratio covered by the sun
    int sample_num = static_cast<int>(width * height * 500 * covered_portion);
    if (sample_num < 100)
        sample_num = 100;

    *data = std::vector<float>(width * height * 3, 0);
    const Vec3 d_value = *radiance / static_cast<float>(sample_num);
    const Vec2 factor = {width * 0.5f * kPiInv, height * kPiInv};

    for (int i = 0; i < sample_num; ++i)
    {
        Vec3 dir = SampleConeUniform(cos_theta, (i + 1.0f) / sample_num,
                                     GetVanDerCorputSequence(i + 1, 2));
        dir = ToWorld(dir, sun_direction);

        float local_azimuth = atan2f(dir.x, -dir.z),
              local_elevation = acosf(fminf(1.0f, fmaxf(-1.0f, dir.y)));
        if (local_azimuth < 0)
            local_azimuth += 2.0f * kPi;

        const int x = std::min(std::max(0, static_cast<int>(local_azimuth * factor.u)), width - 1),
                  y = std::min(std::max(0, static_cast<int>(local_elevation * factor.v)),
                               height - 1);
        const int offset = (x + y * width) * 3;
        for (int c = 0; c < 3; ++c)
            (*data)[offset + c] += d_value[c];
    }
}

void CreateSkyTexture(const Vec3 &sun_direction, const Vec3 &albedo, float turbidity,
                      float stretch, float sky_scale, bool extend, int width, int height,
                      std::vector<float> *data)
{
    *data = std::vector<float>(width * height * 3, 0);
    float zenith = std::acos(std::min(1.0f, std::max(-1.0f, sun_direction.y)));
    float azimuth = std::atan2(sun_direction.x, -sun_direction.z);
    if (azimuth < 0)
        azimuth += 2.0 * kPi;

    ArHosekSkyModelState *skymodel_state[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        skymodel_state[i] = arhosek_rgb_skymodelstate_alloc_init(turbidity, albedo[i],
                                                                 0.5f * kPi - zenith);
    }

    Vec2 factor = {2.0f * kPi / width, kPi / height};
    for (int y = 0; y < height; ++y)
    {
        float theta_raw = (y + 0.5) * factor.v;
        float theta = theta_raw / stretch;
        float factor2 = 1.0;
        if (std::cos(theta) <= 0.0)
        {
            if (extend)
            {
                theta = 0.5f * kPi - kEpsilonFloat;

                float temp1 = 2.0 - 2.0 * theta_raw * kPiInv;
                temp1 = std::min(1.0f, std::max(0.0f, temp1));
                factor2 = temp1 * temp1 * (-2.0f * temp1 + 3);
            }
            else
            {
                continue;
            }
        }

        for (int x = 0; x < width; ++x)
        {
            float phi = (x + .5) * factor.u;
            float cos_gamma = std::cos(theta) * std::cos(zenith) +
                              std::sin(theta) * std::sin(zenith) * std::cos(phi - azimuth);
            float gamma = std::acos(std::min(1.0f, std::max(-1.0f, cos_gamma)));

            Vec3 color;
            for (int i = 0; i < 3; ++i)
            {
                color[i] = static_cast<float>(arhosek_tristim_skymodel_radiance(skymodel_state[i],
                                                                                theta, gamma, i)) /
                           106.856980f;
                color[i] = std::max(0.0f, color[i]);
            }
            color *= sky_scale * factor2;
            const int offset = (x + y * width) * 3;
            for (int i = 0; i < 3; ++i)
                (*data)[offset + i] = color[i];
        }
    }

    for (int i = 0; i < 3; ++i)
        arhosekskymodelstate_free(skymodel_state[i]);
}
