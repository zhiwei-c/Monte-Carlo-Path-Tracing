#ifndef CSRT__RENDERER__BSDF__KULLA_CONTY_HPP
#define CSRT__RENDERER__BSDF__KULLA_CONTY_HPP

#include "../../defs.hpp"

namespace csrt
{

constexpr uint32_t kLutResolution = 128;

void ComputeKullaConty(float *brdf_buffer, float *albedo_avg_buffer);

QUALIFIER_D_H float GetBrdfAvg(float *brdf_avg_buffer, const float cos_theta,
                               const float roughness);
                    
QUALIFIER_D_H float GetAlbedoAvg(float *albedo_avg_buffer, const float roughness);

} // namespace csrt

#endif