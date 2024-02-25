#pragma once

#include "../tensor.cuh"

namespace csrt
{

struct Ray
{
    float t_min;
    float t_max;
#ifdef WATERTIGHT_TRIANGLES
    int k[3];
    Vec3 shear;
#endif
    Vec3 origin;
    Vec3 dir;
    Vec3 dir_rcp;

    QUALIFIER_D_H Ray();
    QUALIFIER_D_H Ray(const Vec3 &_origin, const Vec3 &_dir);

    QUALIFIER_D_H static Vec3 Reflect(const Vec3 &wi, const Vec3 &normal);
    QUALIFIER_D_H static bool Refract(const Vec3 &wi, const Vec3 &normal,
                                      const float eta_inv, Vec3 *wt);
};

} // namespace csrt