#pragma once

#include <map>

#include "../utils/math.h"

NAMESPACE_BEGIN(raytracer)

//光线类
class Ray
{

public:
	/**
	 * \brief 光线
	 * \param origin 起点
	 * \param dir 方向
	 */
	Ray(const Vector3 &origin, const Vector3 &dir) : origin_(origin), dir_(glm::normalize(dir))
	{
		dir_inv_ = Vector3(1.0 / dir_.x, 1.0 / dir_.y, 1.0 / dir_.z);
	}

	///\return 光线方向
	Vector3 dir() const { return dir_; }

	///\return 光线方向的倒数
	Vector3 dir_inv() const { return dir_inv_; }

	///\return 光线起点
	Vector3 origin() const { return origin_; }

private:
	Vector3 origin_;  //光线起点
	Vector3 dir_;	  //光线方向
	Vector3 dir_inv_; //光线方向的倒数
};

/**
 * \brief 根据光线入射方向与表面法线方向，计算光线完美镜面反射方向
 * \param wi 光线入射方向
 * \param normal 表面法线方向（注意：已处理为与光线入射方向夹角大于90度）
 * \return 完美镜面反射方向
 */
inline Vector3 Reflect(const Vector3 &wi, Vector3 normal)
{
	return glm::normalize(wi - static_cast<Float>(2.0 * glm::dot(wi, normal)) * normal);
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算光线完美折射方向
 * \param wi 光线入射方向；
 * \param normal 表面法线方向；（注意：已处理为与光线入射方向夹角大于90度）
 * \param eta_inv 相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
 * \return 光线完美折射方向
 */
inline Vector3 Refract(const Vector3 &wi, const Vector3 &normal, Float eta_inv)
{
	Float cos_theta_i = std::abs(glm::dot(wi, normal)),
		  k = 1.0 - Sqr(eta_inv) * (1.0 - Sqr(cos_theta_i));
	return (k < 0) ? Vector3(0) : glm::normalize((eta_inv * wi + (eta_inv * cos_theta_i - std::sqrt(k)) * normal));
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算菲涅尔系数；
 * \param wi 光线入射方向
 * \param normal 表面法线方向（注意：已处理为与光线入射方向夹角大于90度）
 * \param eta_inv 相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
 * \return 菲涅尔系数。
 */
inline Float Fresnel(const Vector3 &wi, const Vector3 &normal, Float eta_inv)
{
	Float cos_theta_i = std::abs(glm::dot(wi, normal)),
		  cos_theta_t_2 = 1.0 - Sqr(eta_inv) * (1.0 - Sqr(cos_theta_i));
	if (cos_theta_t_2 <= 0)
		return 1;
	else
	{
		Float cos_theta_t = std::sqrt(cos_theta_t_2),
			  Rs_sqrt = (eta_inv * cos_theta_i - cos_theta_t) / (eta_inv * cos_theta_i + cos_theta_t),
			  Rp_sqrt = (cos_theta_i - eta_inv * cos_theta_t) / (cos_theta_i + eta_inv * cos_theta_t);
		return (Rs_sqrt * Rs_sqrt + Rp_sqrt * Rp_sqrt) / 2.0;
	}
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算导体的菲涅尔系数；
 * \param wi 光线入射方向
 * \param normal 表面法线方向
 * \param eta_r 相对折射率的实部
 * \param eta_i 相对折射率的虚部（消光系数）
 * \return 菲涅尔系数
 */
inline Spectrum FresnelConductor(const Vector3 &wi, const Vector3 &normal, const Spectrum &eta_r, const Spectrum &eta_i)
{
	Float cos_theta_i = glm::dot(-wi, normal),
		  cos_theta_i_2 = cos_theta_i * cos_theta_i,
		  sin_theta_i_2 = 1.0 - cos_theta_i_2,
		  sin_theta_i_4 = sin_theta_i_2 * sin_theta_i_2;

	Spectrum temp_1 = eta_r * eta_r - eta_i * eta_i - sin_theta_i_2,
			 a_2_pb_2 = temp_1 * temp_1 + static_cast<Float>(4) * eta_i * eta_i * eta_r * eta_r;
	for (int i = 0; i < 3; i++)
	{
		a_2_pb_2[i] = std::sqrt(std::max(0.0, a_2_pb_2[i]));
	}
	Spectrum a = static_cast<Float>(.5) * (a_2_pb_2 + temp_1);
	for (int i = 0; i < 3; i++)
	{
		a[i] = std::sqrt(std::max(0.0, a[i]));
	}
	Spectrum term_1 = a_2_pb_2 + cos_theta_i_2,
			 term_2 = 2.0 * cos_theta_i * a,
			 term_3 = a_2_pb_2 * cos_theta_i_2 + sin_theta_i_4,
			 term_4 = term_2 * sin_theta_i_2,
			 r_s = (term_1 - term_2) / (term_1 + term_2),
			 r_p = r_s * (term_3 - term_4) / (term_3 + term_4);
	return static_cast<Float>(0.5) * (r_s + r_p);
}

/**
 * \brief Computes the diffuse unpolarized Fresnel reflectance of a dielectric
 *		material (sometimes referred to as "Fdr").
 *		This value quantifies what fraction of diffuse incident illumination
 *		will, on average, be reflected at a dielectric material boundary
 * \param eta Relative refraction coefficient
 * \return F, the unpolarized Fresnel coefficient.
 */
inline Float AverageFresnel(Float eta)
{
	if (eta < 1)
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
		Float inv_eta = 1.0 / eta,
			  inv_eta_2 = inv_eta * inv_eta,
			  inv_eta_3 = inv_eta_2 * inv_eta,
			  inv_eta_4 = inv_eta_3 * inv_eta,
			  inv_eta_5 = inv_eta_4 * inv_eta;
		return 0.919317 - 3.4793 * inv_eta + 6.75335 * inv_eta_2 - 7.80989 * inv_eta_3 + 4.98554 * inv_eta_4 - 1.36881 * inv_eta_5;
	}
}

/**
 * \brief 根据 Gulbrandsen 提出的方法，将金属的折射率 eta 和消光系数 k 重新映射为两个更直观的参数——反射率 reflectivity 和边缘色差 edgetint，
 * 		反射率具体而言是光线垂直表面入射时的反射率，边缘色差控制了观察方向与表面平行时颜色的偏差，
 * 		参见 [《artist Friendly Metallic Fresnel》](https://jcgt.org/published/0003/04/03/paper.pdf)
 * \param eta 相对折射率的实部
 * \param k 相对折射率的虚部（消光系数）
 * \return 由两个 Spectrum 类型构成的 pair，分别代表反射率 reflectivity 和边缘色差 edgetint
 */
inline void IorToReflectivityEdgetint(const Spectrum &eta, const Spectrum &k,
									  Spectrum &reflectivity, Spectrum &edgetint)
{
	Float temp1 = 0, temp2 = 0, temp3 = 0;
	for (int i = 0; i < 3; i++)
	{
		reflectivity[i] = (Sqr(eta[i] - 1) + Sqr(k[i])) / (Sqr(eta[i] + 1.0) + Sqr(k[i]));
		temp1 = 1.0 + std::sqrt(reflectivity[i]);
		temp2 = 1.0 - std::sqrt(reflectivity[i]);
		temp3 = (1.0 - reflectivity[i]) / (1.0 + reflectivity[i]);
		edgetint[i] = (temp1 - eta[i] * temp2) / (temp1 - temp3 * temp2);
	}
}

///\brief 导体材质的平均菲涅尔系数，https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
inline Spectrum AverageFresnelConductor(const Spectrum &r, const Spectrum &g)
{
	return Spectrum(0.087237) + 0.0230685 * g - 0.0864902 * g * g + 0.0774594 * g * g * g + 0.782654 * r - 0.136432 * r * r + 0.278708 * r * r * r + 0.19744 * g * r + 0.0360605 * g * g * r - 0.2586 * g * r * r;
}

NAMESPACE_END(raytracer)