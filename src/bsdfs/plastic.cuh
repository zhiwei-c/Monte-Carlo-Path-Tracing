#include "bsdf.cuh"

class Plastic : public Bsdf
{
public:
    QUALIFIER_DEVICE Plastic(const uint32_t id, const Bsdf::Info::Data &data)
        : Bsdf(id, kPlastic, data.twosided, data.id_opacity, data.id_bumpmap),
          id_roughness_(data.plastic.id_roughness),
          id_diffuse_reflectance_(data.plastic.id_diffuse_reflectance),
          id_specular_reflectance_(data.plastic.id_specular_reflectance),
          reflectivity_(pow(data.plastic.eta - 1.0f, 2) / pow(data.plastic.eta + 1.0f, 2)),
          F_avg_(AverageFresnelDielectric(data.plastic.eta))
    {
    }

    QUALIFIER_DEVICE void Evaluate(Texture **texture_buffer, const float *pixel_buffer,
                                   uint32_t *seed, SamplingRecord *rec) const override;

    QUALIFIER_DEVICE void Sample(Texture **texture_buffer, const float *pixel_buffer,
                                 uint32_t *seed, SamplingRecord *rec) const override;

private:
    float reflectivity_;
    float F_avg_;
    uint32_t id_roughness_;
    uint32_t id_diffuse_reflectance_;
    uint32_t id_specular_reflectance_;
};