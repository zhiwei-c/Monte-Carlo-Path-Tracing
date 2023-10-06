#include "ndf.hpp"

#include "../math/coordinate.hpp"
#include "../math/math.hpp"

NAMESPACE_BEGIN(raytracer)

void GgxNdf::Sample(const dvec3 &n, double alpha_u, double alpha_v, const dvec2 &sample, dvec3 *h, double *pdf) const
{
    double sin_phi = 0.0, cos_phi = 0.0, alpha_2 = 0.0;
    if (alpha_u == alpha_v)
    {
        double phi = 2.0 * kPi * sample.y;
        cos_phi = std::cos(phi);
        sin_phi = std::sin(phi);
        alpha_2 = alpha_u * alpha_u;
    }
    else
    {
        double phi = std::atan(alpha_v / alpha_u * std::tan(kPi + 2.0 * kPi * sample.y)) + kPi * std::floor(2.0 * sample.y + 0.5);
        cos_phi = std::cos(phi);
        sin_phi = std::sin(phi);
        alpha_2 = 1.0 / (Sqr(cos_phi / alpha_u) + Sqr(sin_phi / alpha_v));
    }
    double tan_theta_2 = alpha_2 * sample.x / (1.0 - sample.x),
          cos_theta = 1.0 / std::sqrt(1.0 + tan_theta_2),
          sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    *h = ToWorld({sin_theta * cos_phi, sin_theta * sin_phi, cos_theta}, n);
    *pdf = 1.0 / static_cast<double>(kPi * alpha_u * alpha_v * std::pow(cos_theta, 3) * Sqr(1.0 + tan_theta_2 / alpha_2));
}

double GgxNdf::Pdf(const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const
{
    double cos_theta = glm::dot(h, n);
    if (cos_theta <= 0.0)
    {
        return 0.0;
    }
    double cos_theta_2 = Sqr(cos_theta),
          tan_theta_2 = (1.0 - cos_theta_2) / cos_theta_2,
          alpha_2 = alpha_u * alpha_v;
    if (alpha_u == alpha_v)
    {
        return alpha_2 / static_cast<double>(kPi * std::pow(cos_theta, 3) * Sqr(alpha_2 + tan_theta_2));
    }
    else
    {
        dvec3 dir = ToLocal(h, n);
        return cos_theta / (kPi * alpha_2 * Sqr(Sqr(dir.x / alpha_u) + Sqr(dir.y / alpha_v) + Sqr(dir.z)));
    }
}

double GgxNdf::SmithG1(const dvec3 &v, const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const
{
    double cos_v_n = glm::dot(v, n);
    if (cos_v_n * glm::dot(v, h) <= 0.0)
    {
        return 0.0;
    }

    if (std::abs(cos_v_n - 1.0) < std::numeric_limits<double>::epsilon())
    {
        return 1.0;
    }

    if (alpha_u == alpha_v)
    {
        double cos_v_n_2 = Sqr(cos_v_n),
              tan_v_n_2 = (1.0 - cos_v_n_2) / cos_v_n_2,
              alpha_2 = alpha_u * alpha_u;
        return 2.0 / (1.0 + std::sqrt(1.0 + alpha_2 * tan_v_n_2));
    }
    else
    {
        dvec3 dir = ToLocal(v, n);
        double xy_alpha_2 = Sqr(alpha_u * dir.x) + Sqr(alpha_v * dir.y),
              tan_v_n_alpha_2 = xy_alpha_2 / Sqr(dir.z);
        return 2.0 / (1.0 + std::sqrt(1.0 + tan_v_n_alpha_2));
    }
}

NAMESPACE_END(raytracer)