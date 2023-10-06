#include "ray.hpp"

#include "../math/math.hpp"

NAMESPACE_BEGIN(raytracer)

Ray::Ray(const dvec3 &origin, const dvec3 &direction)
	: origin_(origin),
	  dir_(glm::normalize(direction)),
	  dir_rcp_({1.0 / dir_.x, 1.0 / dir_.y, 1.0 / dir_.z}),
	  t_max_(kMaxDouble)
{
}

Ray::Ray(const dvec3 &origin, const dvec3 &direction, double t_max)
	: origin_(origin),
	  dir_(glm::normalize(direction)),
	  dir_rcp_({1.0 / dir_.x, 1.0 / dir_.y, 1.0 / dir_.z}),
	  t_max_(t_max)
{
}

dvec3 Reflect(const dvec3 &wi, const dvec3 &normal)
{
	return glm::normalize(wi - 2.0 * glm::dot(wi, normal) * normal);
}

dvec3 Refract(const dvec3 &wi, const dvec3 &normal, double eta_inv)
{
	double cos_theta_i = std::abs(glm::dot(wi, normal)),
		   k = 1.0 - Sqr(eta_inv) * (1.0 - Sqr(cos_theta_i));
	return (k < 0.0) ? dvec3(0) : glm::normalize((eta_inv * wi + (eta_inv * cos_theta_i - std::sqrt(k)) * normal));
}

dvec3 AverageFresnelConductor(const dvec3 &eta, const dvec3 &k)
{
	auto reflectivity = dvec3(0),
		 edgetint = dvec3(0);
	double temp1 = 0.0, temp2 = 0.0, temp3 = 0.0;
	for (int i = 0; i < 3; i++)
	{
		reflectivity[i] = (Sqr(eta[i] - 1) + Sqr(k[i])) / (Sqr(eta[i] + 1.0) + Sqr(k[i]));
		temp1 = 1.0 + std::sqrt(reflectivity[i]);
		temp2 = 1.0 - std::sqrt(reflectivity[i]);
		temp3 = (1.0 - reflectivity[i]) / (1.0 + reflectivity[i]);
		edgetint[i] = (temp1 - eta[i] * temp2) / (temp1 - temp3 * temp2);
	}

	return dvec3(0.087237) +
		   0.0230685 * edgetint -
		   0.0864902 * edgetint * edgetint +
		   0.0774594 * edgetint * edgetint * edgetint +
		   0.782654 * reflectivity -
		   0.136432 * reflectivity * reflectivity +
		   0.278708 * reflectivity * reflectivity * reflectivity +
		   0.19744 * edgetint * reflectivity +
		   0.0360605 * edgetint * edgetint * reflectivity -
		   0.2586 * edgetint * reflectivity * reflectivity;
}

/**
 * \brief Computes the diffuse unpolarized Fresnel reflectance of a dielectric
 *		material (sometimes referred to as "Fdr").
 *		This value quantifies what fraction of diffuse incident illumination
 *		will, on average, be reflected at a dielectric material boundary
 * \param eta Relative refraction coefficient
 * \return F, the unpolarized Fresnel coefficient.
 */
double AverageFresnelDielectric(double eta)
{
	if (eta < 1.0)
	{
		/* Fit by Egan and Hilgeman (1973). Works reasonably well for
			"normal" IOR values (<2).
			Max rel. error in 1.0 - 1.5 : 0.1%
			Max rel. error in 1.5 - 2   : 0.6%
			Max rel. error in 2.0 - 5   : 9.5%
		*/
		return -1.4399 * (eta * eta) + 0.7099 * eta + 0.6681 + 0.0636 / eta;
	}
	else
	{
		/* Fit by d'Eon and Irving (2011)

			Maintains a good accuracy even for unrealistic IOR values.

			Max rel. error in 1.0 - 2.0   : 0.1%
			Max rel. error in 2.0 - 10.0  : 0.2%
		*/
		double inv_eta = 1.0 / eta,
			   inv_eta_2 = inv_eta * inv_eta,
			   inv_eta_3 = inv_eta_2 * inv_eta,
			   inv_eta_4 = inv_eta_3 * inv_eta,
			   inv_eta_5 = inv_eta_4 * inv_eta;
		return 0.919317 - 3.4793 * inv_eta + 6.75335 * inv_eta_2 - 7.80989 * inv_eta_3 + 4.98554 * inv_eta_4 - 1.36881 * inv_eta_5;
	}
}

double FresnelDielectric(const dvec3 &wi, const dvec3 &normal, double eta_inv)
{
	double cos_theta_i = std::abs(glm::dot(wi, normal)),
		   cos_theta_t_2 = 1.0 - Sqr(eta_inv) * (1.0 - Sqr(cos_theta_i));
	if (cos_theta_t_2 <= 0.0)
	{
		return 1.0;
	}
	else
	{
		double cos_theta_t = std::sqrt(cos_theta_t_2),
			   Rs_sqrt = (eta_inv * cos_theta_i - cos_theta_t) / (eta_inv * cos_theta_i + cos_theta_t),
			   Rp_sqrt = (cos_theta_i - eta_inv * cos_theta_t) / (cos_theta_i + eta_inv * cos_theta_t);
		return (Rs_sqrt * Rs_sqrt + Rp_sqrt * Rp_sqrt) / 2.0;
	}
}

dvec3 FresnelConductor(const dvec3 &wi, const dvec3 &normal, const dvec3 &eta_r, const dvec3 &eta_i)
{
	double cos_theta_i = glm::dot(-wi, normal),
		   cos_theta_i_2 = cos_theta_i * cos_theta_i,
		   sin_theta_i_2 = 1.0 - cos_theta_i_2,
		   sin_theta_i_4 = sin_theta_i_2 * sin_theta_i_2;

	dvec3 temp_1 = eta_r * eta_r - eta_i * eta_i - sin_theta_i_2,
		  a_2_pb_2 = temp_1 * temp_1 + 4.0 * eta_i * eta_i * eta_r * eta_r;
	for (int i = 0; i < 3; i++)
	{
		a_2_pb_2[i] = std::sqrt(std::max(0.0, a_2_pb_2[i]));
	}
	dvec3 a = 0.5 * (a_2_pb_2 + temp_1);
	for (int i = 0; i < 3; i++)
	{
		a[i] = std::sqrt(std::max(0.0, a[i]));
	}
	dvec3 term_1 = a_2_pb_2 + cos_theta_i_2,
		  term_2 = 2.0 * cos_theta_i * a,
		  term_3 = a_2_pb_2 * cos_theta_i_2 + sin_theta_i_4,
		  term_4 = term_2 * sin_theta_i_2,
		  r_s = (term_1 - term_2) / (term_1 + term_2),
		  r_p = r_s * (term_3 - term_4) / (term_3 + term_4);
	return 0.5 * (r_s + r_p);
}

NAMESPACE_END(raytracer)