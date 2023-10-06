#pragma once

#include <array>

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//光线
class Ray
{
public:
    Ray(const dvec3 &origin, const dvec3 &direction);
    Ray(const dvec3 &origin, const dvec3 &direction, double t_max);

    const double &t_max() const { return t_max_; }
    const dvec3 &origin() const { return origin_; }
    const dvec3 &dir() const { return dir_; }
    const dvec3 &dir_rcp() const { return dir_rcp_; }

private:
    double t_max_;  //最大传播距离
    dvec3 origin_;  //光线起点
    dvec3 dir_;     //光线方向
    dvec3 dir_rcp_; //光线方向的倒数
};

dvec3 Reflect(const dvec3 &wi, const dvec3 &normal);
dvec3 Refract(const dvec3 &wi, const dvec3 &normal, double eta_inv);

double AverageFresnelDielectric(double eta);
dvec3 FresnelConductor(const dvec3 &wi, const dvec3 &normal, const dvec3 &eta_r, const dvec3 &eta_i);

double FresnelDielectric(const dvec3 &wi, const dvec3 &normal, double eta_inv);
dvec3 AverageFresnelConductor(const dvec3 &eta, const dvec3 &k);

NAMESPACE_END(raytracer)