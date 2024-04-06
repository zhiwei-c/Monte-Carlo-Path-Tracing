#ifndef CSRT__RENDERER__INTEGRATORS__PATH_HPP
#define CSRT__RENDERER__INTEGRATORS__PATH_HPP

#include "../../rtcore/scene.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../bsdfs/bsdf.hpp"
#include "../emitters/emitter.hpp"
#include "../medium/medium.hpp"

namespace csrt
{

struct IntegratorData;

QUALIFIER_D_H Vec3 ShadePath(const IntegratorData *data, const Vec3 &eye,
                             const Vec3 &look_dir, uint32_t *seed);

QUALIFIER_D_H Vec3 EvaluateDirectLightPath(const IntegratorData *data,
                                           const Hit &hit, const Vec3 &wo,
                                           uint32_t *seed);

QUALIFIER_D_H BsdfSampleRec EvaluateRayPath(const Vec3 &wi, const Vec3 &wo,
                                            const Hit &hit, Bsdf *bsdf);

QUALIFIER_D_H BsdfSampleRec SampleRayPath(const Vec3 &wo, const Hit &hit,
                                          Bsdf *bsdf, uint32_t *seed);

} // namespace csrt

#endif