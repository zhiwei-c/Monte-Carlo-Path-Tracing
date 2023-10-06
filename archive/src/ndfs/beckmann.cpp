#include "ndf.hpp"

#include "../math/coordinate.hpp"
#include "../math/math.hpp"

NAMESPACE_BEGIN(raytracer)

void BeckmannNdf::Sample(const dvec3 &n, double alpha_u, double alpha_v, const dvec2 &sample, dvec3 *h, double *pdf) const
{
    double sin_phi = 0.0, cos_phi = 0.0, alpha_2 = 0.0;
    if (alpha_u == alpha_v)
    {
        double phi = 2.0 * kPi * sample.y;
        cos_phi = std::cos(phi);
        sin_phi = std::sin(phi);
        alpha_2 = alpha_u * alpha_v;
    }
    else
    {
        double phi = std::atan(alpha_v / alpha_u * std::tan(kPi + 2.0 * kPi * sample.y)) + kPi * std::floor(2.0 * sample.y + 0.5);
        cos_phi = std::cos(phi);
        sin_phi = std::sin(phi);
        alpha_2 = 1.0 / (Sqr(cos_phi / alpha_u) + Sqr(sin_phi / alpha_v));
    }

    double cos_theta = 1.0 / std::sqrt(1.0 - alpha_2 * std::log(1.0 - sample.x)),
          sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    *h = ToWorld({sin_theta * cos_phi, sin_theta * sin_phi, cos_theta}, n);
    *pdf = (1.0 - sample.x) / static_cast<double>(kPi * alpha_u * alpha_v * std::pow(cos_theta, 3));
}

double BeckmannNdf::Pdf(const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const
{
    double cos_theta = glm::dot(h, n);
    if (cos_theta <= 0.0)
    {
        return 0.0;
    }
    double cos_theta_2 = Sqr(cos_theta),
          tan_theta_2 = (1.0 - cos_theta_2) / cos_theta_2,
          cos_theta_3 = cos_theta_2 * cos_theta,
          alpha_2 = alpha_u * alpha_v;
    if (alpha_u == alpha_v)
    {
        return std::exp(-tan_theta_2 / alpha_2) / (kPi * alpha_2 * cos_theta_3);
    }
    else
    {
        dvec3 dir = ToLocal(h, n);
        return std::exp(-(Sqr(dir.x / alpha_u) + Sqr(dir.y / alpha_v)) / cos_theta_2) / (kPi * alpha_2 * cos_theta_3);
    }
}

double BeckmannNdf::SmithG1(const dvec3 &v, const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const
{
    double cos_v_n = glm::dot(v, n);
    if (cos_v_n * glm::dot(v, h) <= 0.0)
    {
        return 0.0;
    }

    if (std::abs(cos_v_n - 1.0) <= 0.0)
    {
        return 1.0;
    }

    double a = 0.0;
    if (alpha_u == alpha_v)
    {
        const double tan_theta_v_n = std::sqrt(1.0 - Sqr(cos_v_n)) / cos_v_n;
        a = 1.0 / (alpha_u * tan_theta_v_n);
    }
    else
    {
        const dvec3 dir = ToLocal(v, n);
        const double xy_alpha_2 = Sqr(alpha_u * dir.x) + Sqr(alpha_v * dir.y),
                    tan_theta_alpha_2 = xy_alpha_2 / Sqr(dir.z);
        a = 1.0 / std::sqrt(tan_theta_alpha_2);
    }

    if (a < 1.6)
    {
        double a_2 = a * a;
        return (3.535 * a + 2.181 * a_2) / (1.0 + 2.276 * a + 2.577 * a_2);
    }
    else
    {
        return 1.0;
    }
}

NAMESPACE_END(raytracer)