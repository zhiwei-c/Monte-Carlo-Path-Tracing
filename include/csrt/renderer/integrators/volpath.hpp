#ifndef CSRT__RENDERER__INTEGRATORS__VOLPATH_HPP
#define CSRT__RENDERER__INTEGRATORS__VOLPATH_HPP

#include "path.hpp"

namespace csrt
{

struct IntegratorData;

struct MediumHit
{
    Vec3 position = {};
    Medium *medium = nullptr;
};

QUALIFIER_D_H Vec3 ShadeVolPath(const IntegratorData *data, const Vec3 &eye,
                                const Vec3 &look_dir, uint32_t *seed);

QUALIFIER_D_H Vec3 EvaluateDirectLightVolPath(const IntegratorData *data,
                                              const Hit &hit, const Vec3 &wo,
                                              uint32_t *seed);

QUALIFIER_D_H Vec3 EvaluateDirectLightVolPath(const IntegratorData *data,
                                              const MediumHit &hit, const Vec3 &wo,
                                              uint32_t *seed);

} // namespace csrt

#endif