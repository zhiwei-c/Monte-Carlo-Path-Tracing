#pragma once

#include <map>

#include "../utils/global.h"

class Ray
{
public:
    /**
     * \brief 光线
     * \param origin 起点
     * \param dir 方向
     */
    __device__ Ray(const vec3 &origin, const vec3 &dir) : origin_(origin), dir_(myvec::normalize(dir))
    {
        dir_inv_ = vec3(1 / dir_.x, 1 / dir_.y, 1 / dir_.z);
    }

    ///\return 光线方向
    __device__ vec3 dir() const { return dir_; }

    ///\return 光线方向的倒数
    __device__ vec3 dir_inv() const { return dir_inv_; }

    ///\return 光线起点
    __device__ vec3 origin() const { return origin_; }

private:
    vec3 origin_;  //光线起点
    vec3 dir_;     //光线方向
    vec3 dir_inv_; //光线方向的倒数
};

/**
 * \brief 根据光线入射方向与表面法线方向，计算光线完美镜面反射方向
 * \param wi 光线入射方向
 * \param normal 表面法线方向（注意：已处理为与光线入射方向夹角大于90度）
 * \return 完美镜面反射方向
 */
__host__ __device__ inline vec3 Reflect(const vec3 &wi, vec3 normal)
{
    return myvec::normalize(wi - static_cast<Float>(2 * myvec::dot(wi, normal)) * normal);
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算光线完美折射方向
 * \param wi 光线入射方向；
 * \param normal 表面法线方向；（注意：已处理为与光线入射方向夹角大于90度）
 * \param eta_inv 相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
 * \return 光线完美折射方向
 */
__device__ inline vec3 Refract(const vec3 &wi, const vec3 &normal, Float eta_inv)
{
    auto cos_theta_i = abs(myvec::dot(wi, normal));
    auto k = 1 - eta_inv * eta_inv * (1 - cos_theta_i * cos_theta_i);
    return (k < 0) ? vec3(0) : myvec::normalize((eta_inv * wi + (eta_inv * cos_theta_i - sqrt(k)) * normal));
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算菲涅尔系数；
 * \param wi 光线入射方向
 * \param normal 表面法线方向（注意：已处理为与光线入射方向夹角大于90度）
 * \param eta_inv 相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
 * \return 菲涅尔系数。
 */
__device__ inline Float Fresnel(const vec3 &wi, const vec3 &normal, Float eta_inv)
{
    auto cos_theta_i = abs(myvec::dot(wi, normal));
    auto cos_theta_t_2 = 1.0 - eta_inv * eta_inv * (1.0 - cos_theta_i * cos_theta_i);

    if (cos_theta_t_2 <= 0)
    {
        return 1;
    }
    else
    {
        auto cos_theta_t = sqrt(cos_theta_t_2);
        auto Rs_sqrt = (eta_inv * cos_theta_i - cos_theta_t) / (eta_inv * cos_theta_i + cos_theta_t),
             Rp_sqrt = (cos_theta_i - eta_inv * cos_theta_t) / (cos_theta_i + eta_inv * cos_theta_t);
        return (Rs_sqrt * Rs_sqrt + Rp_sqrt * Rp_sqrt) / 2;
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
__device__ inline vec3 FresnelConductor(const vec3 &wi, const vec3 &normal, const vec3 &eta_r, const vec3 &eta_i)
{
    auto cos_theta_i = myvec::dot(-wi, normal);
    auto cos_theta_i_2 = cos_theta_i * cos_theta_i,
         sin_theta_i_2 = 1.0 - cos_theta_i_2,
         sin_theta_i_4 = sin_theta_i_2 * sin_theta_i_2;

    auto temp_1 = eta_r * eta_r - eta_i * eta_i - sin_theta_i_2;

    auto a_2_pb_2 = temp_1 * temp_1 + static_cast<Float>(4) * eta_i * eta_i * eta_r * eta_r;
    for (int i = 0; i < 3; i++)
    {
        a_2_pb_2[i] = sqrt(glm::max(0.0, a_2_pb_2[i]));
    }

    auto a = static_cast<Float>(.5) * (a_2_pb_2 + temp_1);
    for (int i = 0; i < 3; i++)
    {
        a[i] = sqrt(glm::max(0.0, a[i]));
    }

    auto term_1 = a_2_pb_2 + cos_theta_i_2,
         term_2 = 2.0 * cos_theta_i * a;

    auto r_s = (term_1 - term_2) / (term_1 + term_2);

    auto term_3 = a_2_pb_2 * cos_theta_i_2 + sin_theta_i_4,
         term_4 = term_2 * sin_theta_i_2;

    auto r_p = r_s * (term_3 - term_4) / (term_3 + term_4);

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
__device__ inline Float AverageFresnel(Float eta)
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
        auto inv_eta = 1.0 / eta,
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
__device__ inline void IorToReflectivityEdgetint(const vec3 &eta,
                                                 const vec3 &k,
                                                 vec3 &reflectivity,
                                                 vec3 &edgetint)
{
    Float temp1, temp2, temp3;
    for (int i = 0; i < 3; i++)
    {
        reflectivity[i] = (pow(eta[i] - 1, 2) + pow(k[i], 2)) / (pow(eta[i] + 1, 2) + pow(k[i], 2));

        temp1 = 1.0 + sqrt(reflectivity[i]);
        temp2 = 1.0 - sqrt(reflectivity[i]);
        temp3 = (1.0 - reflectivity[i]) / (1.0 + reflectivity[i]);
        edgetint[i] = (temp1 - eta[i] * temp2) / (temp1 - temp3 * temp2);
    }
}

///\brief 导体材质的平均菲涅尔系数，https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
__device__ inline vec3 AverageFresnelConductor(const vec3 &r, const vec3 &g)
{
    return vec3(0.087237) + 0.0230685 * g - 0.0864902 * g * g + 0.0774594 * g * g * g + 0.782654 * r - 0.136432 * r * r + 0.278708 * r * r * r + 0.19744 * g * r + 0.0360605 * g * g * r - 0.2586 * g * r * r;
}

