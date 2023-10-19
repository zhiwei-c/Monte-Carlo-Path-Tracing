#include "bsdf.cuh"

class ThinDielectric : public Bsdf
{
public:
    QUALIFIER_DEVICE ThinDielectric(const uint32_t id, const Bsdf::Info::Data &data)
        : Bsdf(id, kThinDielectric, data.twosided, data.id_opacity, data.id_bumpmap),
          eta_(data.dielectric.eta), eta_inv_(1.0f / data.dielectric.eta),
          id_roughness_(data.dielectric.id_roughness),
          id_specular_reflectance_(data.dielectric.id_specular_reflectance),
          id_specular_transmittance_(data.dielectric.id_specular_transmittance),
          reflectivity_(pow(data.dielectric.eta - 1.0f, 2) / pow(data.dielectric.eta + 1.0f, 2))
    {
    }

    QUALIFIER_DEVICE void Evaluate(Texture **texture_buffer, const float *pixel_buffer,
                                   uint32_t *seed, SamplingRecord *rec) const override;

    QUALIFIER_DEVICE void Sample(Texture **texture_buffer, const float *pixel_buffer,
                                 uint32_t *seed, SamplingRecord *rec) const override;

private:
    float eta_;     // 相对折射率，即透射侧介质与入射侧介质的绝对折射率之比
    float eta_inv_; // 相对折射率的倒数
    float reflectivity_;
    uint32_t id_roughness_;
    uint32_t id_specular_reflectance_;
    uint32_t id_specular_transmittance_;
};