#include "bsdf.cuh"

class Conductor : public Bsdf
{
public:
    QUALIFIER_DEVICE Conductor(const uint32_t id, const Bsdf::Info::Data &data)
        : Bsdf(id, kConductor, data.twosided, data.id_opacity, data.id_bumpmap),
          id_roughness_(data.conductor.id_roughness),
          id_specular_reflectance_(data.conductor.id_specular_reflectance),
          reflectivity_(data.conductor.reflectivity),
          F_avg_(AverageFresnelConductor(data.conductor.reflectivity, data.conductor.edgetint))
    {
    }

    QUALIFIER_DEVICE void Evaluate(Texture **texture_buffer, const float *pixel_buffer,
                                   uint32_t *seed, SamplingRecord *rec) const override;

    QUALIFIER_DEVICE void Sample(Texture **texture_buffer, const float *pixel_buffer,
                                 uint32_t *seed, SamplingRecord *rec) const override;

private:
    QUALIFIER_DEVICE Vec3 EvaluateMultipleScatter(const float N_dot_I, const float N_dot_O,
                                                  const float roughness) const;

    uint32_t id_roughness_;
    uint32_t id_specular_reflectance_;
    Vec3 reflectivity_;
    Vec3 F_avg_;
};