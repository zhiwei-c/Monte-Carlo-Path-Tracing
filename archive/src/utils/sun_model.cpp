#include "sun_model.hpp"

#include <algorithm>

#include "../core/spectrum.hpp"
#include "../math/math.hpp"

NAMESPACE_BEGIN(raytracer)

static constexpr double kEarthMeanRadius = 6371.01;    // In km
static constexpr double kAstronomicalUnit = 149597890; // In km

/* The following is from the implementation of "A Practical Analytic Model for
   Daylight" by A.J. Preetham, Peter Shirley, and Brian Smits */

/* All data lifted from MI. Units are either [] or cm^-1. refer when in doubt MI */

// k_o Spectrum table from pg 127, MI.
static constexpr double kKoWavelengths[64] = {
    300, 305, 310, 315, 320, 325, 330, 335, 340, 345,
    350, 355, 445, 450, 455, 460, 465, 470, 475, 480,
    485, 490, 495, 500, 505, 510, 515, 520, 525, 530,
    535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
    585, 590, 595, 600, 605, 610, 620, 630, 640, 650,
    660, 670, 680, 690, 700, 710, 720, 730, 740, 750,
    760, 770, 780, 790};

static constexpr double kKoAmplitudes[64] = {
    10.0, 4.8, 2.7, 1.35, .8, .380, .160, .075, .04, .019, .007,
    .0, .003, .003, .004, .006, .008, .009, .012, .014, .017,
    .021, .025, .03, .035, .04, .045, .048, .057, .063, .07,
    .075, .08, .085, .095, .103, .110, .12, .122, .12, .118,
    .115, .12, .125, .130, .12, .105, .09, .079, .067, .057,
    .048, .036, .028, .023, .018, .014, .011, .010, .009,
    .007, .004, .0, .0};

// k_g Spectrum table from pg 130, MI.
static constexpr double kKgWavelengths[4] = {
    759, 760, 770, 771};

static constexpr double kKgAmplitudes[4] = {
    0, 3.0, 0.210, 0};

// k_wa Spectrum table from pg 130, MI.
static constexpr double kKwaWavelengths[13] = {
    689, 690, 700, 710, 720,
    730, 740, 750, 760, 770,
    780, 790, 800};

static constexpr double kKwaAmplitudes[13] = {
    0, 0.160e-1, 0.240e-1, 0.125e-1,
    0.100e+1, 0.870, 0.610e-1, 0.100e-2,
    0.100e-4, 0.100e-4, 0.600e-3,
    0.175e-1, 0.360e-1};

/* Wavelengths corresponding to the table below */
static constexpr double kSolarWavelengths[38] = {
    380, 390, 400, 410, 420, 430, 440, 450,
    460, 470, 480, 490, 500, 510, 520, 530,
    540, 550, 560, 570, 580, 590, 600, 610,
    620, 630, 640, 650, 660, 670, 680, 690,
    700, 710, 720, 730, 740, 750};

/* Solar amplitude in watts / (m^2 * nm * sr) */
static constexpr double kSolarAmplitudes[38] = {
    16559.0, 16233.7, 21127.5, 25888.2, 25829.1,
    24232.3, 26760.5, 29658.3, 30545.4, 30057.5,
    30663.7, 28830.4, 28712.1, 27825.0, 27100.6,
    27233.6, 26361.3, 25503.8, 25060.2, 25311.6,
    25355.9, 25134.2, 24631.5, 24173.2, 23685.3,
    23212.1, 22827.7, 22339.8, 21970.2, 21526.7,
    21097.9, 20728.3, 20240.4, 19870.8, 19427.2,
    19072.4, 18628.9, 18259.2};

LocationDataInfo::LocationDataInfo()
    : year(2010),
      month(7),
      day(10),
      hour(15.0),
      minute(0.0),
      second(0.0),
      timezone(9),
      latitude(35.6894),
      longitude(139.6917)
{
}

dvec3 GetSunDirection(const LocationDataInfo &location_time)
{
    // Main variables
    double elapsed_Julian_days, dec_hours;
    double ecliptic_longitude, ecliptic_obliquity;
    double right_scension, declination;
    double zenith, azimuth;

    // Auxiliary variables
    double dy;
    double dx;

    /* Calculate difference in days between the current Julian Day
       and JD 2451545.0, which is noon 1 January 2000 Universal Time */
    {
        // Calculate time of the day in UT decimal hours
        dec_hours = location_time.hour - location_time.timezone +
                    (location_time.minute + location_time.second / 60.0) / 60.0;

        // Calculate current Julian Day
        int li_aux1 = (location_time.month - 14) / 12;
        int li_aux2 = (1461 * (location_time.year + 4800 + li_aux1)) / 4 +
                      (367 * (location_time.month - 2 - 12 * li_aux1)) / 12 -
                      (3 * ((location_time.year + 4900 + li_aux1) / 100)) / 4 +
                      location_time.day - 32075;
        double d_Julian_date = li_aux2 - 0.5 + dec_hours / 24.0;

        // Calculate difference between current Julian Day and JD 2451545.0
        elapsed_Julian_days = d_Julian_date - 2451545.0;
    }

    /* Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
       ecliptic in radians but without limiting the angle to be less than 2*Pi
       (i.e., the result may be greater than 2*Pi) */
    {
        double omega = 2.1429 - 0.0010394594 * elapsed_Julian_days;
        double mean_longitude = 4.8950630 + 0.017202791698 * elapsed_Julian_days; // Radians
        double anomaly = 6.2400600 + 0.0172019699 * elapsed_Julian_days;

        ecliptic_longitude = mean_longitude + 0.03341607 * std::sin(anomaly) + 0.00034894 * std::sin(2 * anomaly) - 0.0001134 - 0.0000203 * std::sin(omega);

        ecliptic_obliquity = 0.4090928 - 6.2140e-9 * elapsed_Julian_days + 0.0000396 * std::cos(omega);
    }

    /* Calculate celestial coordinates ( right ascension and declination ) in radians
       but without limiting the angle to be less than 2*Pi (i.e., the result may be
       greater than 2*Pi) */
    {
        double sin_eclipticLongitude = std::sin(ecliptic_longitude);
        dy = std::cos(ecliptic_obliquity) * sin_eclipticLongitude;
        dx = std::cos(ecliptic_longitude);
        right_scension = std::atan2(dy, dx);
        if (right_scension < 0.0)
        {
            right_scension += 2 * kPi;
        }
        declination = std::asin(std::sin(ecliptic_obliquity) * sin_eclipticLongitude);
    }

    // Calculate local coordinates (azimuth and zenith angle) in degrees
    {
        double greenwich_Mean_sidereal_time = 6.6974243242 + 0.0657098283 * elapsed_Julian_days + dec_hours;

        double local_mean_sidereal_time = glm::radians(greenwich_Mean_sidereal_time * 15 + location_time.longitude);

        double latitude_in_radians = glm::radians(location_time.latitude);
        double cos_latitude = std::cos(latitude_in_radians);
        double sin_latitude = std::sin(latitude_in_radians);

        double hour_angle = local_mean_sidereal_time - right_scension;
        double cos_hour_angle = std::cos(hour_angle);

        zenith = std::acos(cos_latitude * cos_hour_angle * std::cos(declination) + std::sin(declination) * sin_latitude);

        dy = -std::sin(hour_angle);
        dx = std::tan(declination) * cos_latitude - sin_latitude * cos_hour_angle;

        azimuth = std::atan2(dy, dx);
        if (azimuth < 0.0)
        {
            azimuth += 2 * kPi;
        }

        // Parallax Correction
        zenith += (kEarthMeanRadius / kAstronomicalUnit) * std::sin(zenith);
    }

    return dvec3{std::sin(azimuth) * std::sin(zenith),
                 std::cos(zenith),
                 -std::cos(azimuth) * std::sin(zenith)};
}

dvec3 GetSunRadiance(double theta, double turbidity)
{
    static std::vector<double> k_o_wavelengths = {kKoWavelengths, kKoWavelengths + sizeof(kKoWavelengths) / sizeof(kKoWavelengths[0])},
                               k_o_amplitudes = {kKoAmplitudes, kKoAmplitudes + sizeof(kKoAmplitudes) / sizeof(kKoAmplitudes[0])},
                               k_g_wavelengths = {kKgWavelengths, kKgWavelengths + sizeof(kKgWavelengths) / sizeof(kKgWavelengths[0])},
                               K_g_values = {kKgAmplitudes, kKgAmplitudes + sizeof(kKgAmplitudes) / sizeof(kKgAmplitudes[0])},
                               wa_wavelengths = {kKwaWavelengths, kKwaWavelengths + sizeof(kKwaWavelengths) / sizeof(kKwaWavelengths[0])},
                               wa_amplitudes = {kKwaAmplitudes, kKwaAmplitudes + sizeof(kKwaAmplitudes) / sizeof(kKwaAmplitudes[0])},
                               solar_wavelengths = {kSolarWavelengths, kSolarWavelengths + sizeof(kSolarWavelengths) / sizeof(kSolarWavelengths[0])},
                               solar_amplitudes = {kSolarAmplitudes, kSolarAmplitudes + sizeof(kSolarAmplitudes) / sizeof(kSolarAmplitudes[0])};

    auto data = std::vector<double>(91),
         wavelengths = std::vector<double>(91); // (800 - 350) / 5  + 1
    double beta = 0.04608365822050f * turbidity - 0.04586025928522f;

    // Relative Optical Mass
    double m = 1.0f / (std::cos(theta) + 0.15f * std::pow(93.885 - theta / kPi * 180.0, -1.253f));

    double lambda;
    int i = 0;
    for (i = 0, lambda = 350; i < 91; ++i, lambda += 5)
    {
        // Rayleigh Scattering
        // Results agree with the graph (pg 115, MI) */
        double tau_rR = std::exp(-m * 0.008735 * std::pow(lambda / 1000.0, -4.08));

        // Aerosol (water + dust) attenuation
        // beta - amount of aerosols present
        // alpha - ratio of small to large particle sizes. (0:4,usually 1.3)
        // Results agree with the graph (pg 121, MI)
        const double alpha = 1.3;
        double tau_a = std::exp(-m * beta * std::pow(lambda / 1000.0, -alpha)); // lambda should be in um

        // Attenuation due to ozone absorption
        // l_ozone - amount of ozone in cm(NTP)
        // Results agree with the graph (pg 128, MI)
        const double l_ozone = 0.35;
        double tau_o = std::exp(-m * EvalSpectrumAmplitude(k_o_wavelengths, k_o_amplitudes, lambda) * l_ozone);

        // Attenuation due to mixed gases absorption
        // Results agree with the graph (pg 131, MI)
        double tau_g = std::exp(-1.41 * EvalSpectrumAmplitude(k_g_wavelengths, K_g_values, lambda) * m /
                                std::pow(1 + 118.93f * EvalSpectrumAmplitude(k_g_wavelengths, K_g_values, lambda) * m, 0.45));

        // Attenuation due to water vapor absorbtion
        // w - precipitable water vapor in centimeters (standard = 2)
        // Results agree with the graph (pg 132, MI)
        const double w = 2.0;
        double tau_va = std::exp(-0.2385 * EvalSpectrumAmplitude(wa_wavelengths, wa_amplitudes, lambda) * w * m /
                                 std::pow(1 + 20.07 * EvalSpectrumAmplitude(wa_wavelengths, wa_amplitudes, lambda) * w * m, 0.45));

        wavelengths[i] = lambda;
        data[i] = EvalSpectrumAmplitude(solar_wavelengths, solar_amplitudes, lambda) * tau_rR * tau_a * tau_o * tau_g * tau_va;
    }

    dvec3 rgb = SpectrumToRgb(wavelengths, data);
    return rgb;
}

NAMESPACE_END(raytracer)