#include "diffuse.hpp"

#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

Diffuse::Diffuse(const std::string &id, Texture *reflectance)
    : Bsdf(BsdfType::kDiffuse, id),
      reflectance_(reflectance)
{
}

void Diffuse::Sample(SamplingRecord *rec, Sampler *sampler) const
{
    rec->pdf = SampleHemisCos(rec->normal, &rec->wi, sampler->Next2D());
    if (rec->pdf == 0.0)
    {
        return;
    }
    rec->type = ScatteringType::kReflect;
    rec->attenuation = reflectance_->color(rec->texcoord) * kPiRcp * glm::dot(-rec->wi, rec->normal);
}

void Diffuse::Eval(SamplingRecord *rec) const
{
    if (glm::dot(rec->wo, rec->normal) < 0)
    { //表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
        //又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
        //故只需确保光线出射方向和表面法线方向在介质同侧即可
        return;
    }
    rec->pdf = PdfHemisCos(rec->wo, rec->normal);
    if (rec->pdf <= kEpsilonPdf)
    {
        return;
    }
    rec->type = ScatteringType::kReflect;
    rec->attenuation = reflectance_->color(rec->texcoord) * kPiRcp * glm::dot(-rec->wi, rec->normal);
}

bool Diffuse::IsTransparent(const dvec2 &texcoord, Sampler *sampler) const
{
    return Bsdf::IsTransparent(texcoord, sampler) || reflectance_ && reflectance_->IsTransparent(texcoord, sampler);
}

bool Diffuse::UseTextureMapping() const
{
    return Bsdf::UseTextureMapping() || !reflectance_->IsConstant();
}

NAMESPACE_END(raytracer)