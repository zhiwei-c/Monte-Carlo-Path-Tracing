#include "bsdf.hpp"

#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

Bsdf::Bsdf(BsdfType type, const std::string &id)
    : type_(type),
      id_(id),
      twosided_(false),
      is_emitter_(false),
      opacity_(nullptr),
      bump_map_(nullptr),
      radiance_(dvec3(0))
{
}

void Bsdf::ApplyBumpMapping(const dvec3 &tangent, const dvec3 &bitangent, const dvec2 &texcoord, dvec3 *normal) const
{
    if (bump_map_ == nullptr)
    {
        return;
    }

    dvec2 gradient = bump_map_->gradient(texcoord);
    dvec3 normal_pertubed_local = glm::normalize(dvec3{-gradient.x, -gradient.y, 1});
    dmat3 TBN = {tangent, bitangent, *normal};
    *normal = glm::normalize(TBN * normal_pertubed_local);
}

bool Bsdf::IsTransparent(const dvec2 &texcoord, Sampler *sampler) const
{
    return opacity_ && sampler->Next1D() > opacity_->color(texcoord).x;
}

bool Bsdf::IsHarshLobe() const
{
    return type_ == BsdfType::kConductor || type_ == BsdfType::kDielectric || type_ == BsdfType::kThinDielectric;
}

bool Bsdf::UseTextureMapping() const
{
    return opacity_ && !opacity_->IsConstant() ||
           bump_map_ && !bump_map_->IsConstant();
}

void Bsdf::SetRadiance(const dvec3 &radiance)
{
    is_emitter_ = true;
    radiance_ = radiance;
}

NAMESPACE_END(raytracer)