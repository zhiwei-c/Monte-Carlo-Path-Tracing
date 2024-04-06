#include "spot_light.hpp"

#include "../accelerators/accelerator.hpp"
#include "../core/ray.hpp"
#include "../math/coordinate.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

SpotLight::SpotLight(const dvec3 &intensity, double cutoff_angle, double beam_width, Texture *texture, const dmat4 &to_world)
    : Emitter(EmitterType::kSpot),
      intensity_(intensity),
      cutoff_angle_(cutoff_angle),
      cos_cutoff_angle_(std::cos(cutoff_angle)),
      uv_factor_(std::tan(cutoff_angle)),
      cos_beam_width_(std::cos(beam_width)),
      transition_width_rcp_(1.0 / (cutoff_angle - beam_width)),
      texture_(texture),
      to_local_(glm::inverse(to_world)),
      position_world_(TransfromPoint(to_world, {0, 0, 0}))
{
}

SamplingRecord SpotLight::Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                                 Accelerator *accelerator) const
{
    auto rec = SamplingRecord();
    rec.position = its_shape.position();
    rec.wo = wo;
    rec.distance = glm::length(position_world_ - rec.position);
    if (rec.distance < kEpsilonDistance)
    {
        return rec;
    }

    rec.wi = glm::normalize(its_shape.position() - position_world_);
    const Ray ray = Ray(its_shape.position(), -rec.wi, rec.distance);
    auto its_test = Intersection();
    if (accelerator && accelerator->Intersect(ray, sampler, &its_test) &&
        its_test.distance() + kEpsilonDistance < rec.distance)
    {
        return rec;
    }

    auto fall_off = dvec3(0);
    if (!FallOffCurve(TransfromVec(to_local_, rec.wi), &fall_off))
    {
        return rec;
    }
    const double distance_rcp = 1.0 / rec.distance;
    rec.type = ScatteringType::kScattering;
    rec.radiance = intensity_ * fall_off * distance_rcp * distance_rcp;
    rec.pdf = 1;
    return rec;
}

bool SpotLight::FallOffCurve(const dvec3 &local_dir, dvec3 *value) const
{
    const double cos_theta = local_dir.z;
    if (cos_theta <= cos_cutoff_angle_)
    {
        return false;
    }

    *value = dvec3(1.0f);
    if (texture_)
    {
        const dvec2 uv = {0.5 + 0.5 * local_dir.x / (local_dir.z * uv_factor_),
                          0.5 + 0.5 * local_dir.y / (local_dir.z * uv_factor_)};
        *value *= texture_->color(uv);
    }

    if (cos_theta < cos_beam_width_)
    {
        *value *= (cutoff_angle_ - std::acos(cos_theta)) * transition_width_rcp_;
    }

    return true;
}

NAMESPACE_END(raytracer)