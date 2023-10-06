#include "intersection.cuh"

#include "../utils/math.cuh"
#include "../bsdfs/bsdfs.cuh"

QUALIFIER_DEVICE Intersection::Intersection()
    : valid_(false), absorb_(false), inside_(false),
      id_instance_(kInvalidId), id_bsdf_(kInvalidId),
      pos_(Vec3(0)), normal_(Vec3(0)), texcoord_(Vec2(0)),
      distance_(kMaxFloat), pdf_area_(kMaxFloat)

{
}

QUALIFIER_DEVICE Intersection::Intersection(const Vec3 &pos)
    : valid_(true), absorb_(false), inside_(false),
      id_instance_(kInvalidId), id_bsdf_(kInvalidId),
      pos_(pos), normal_(Vec3(0)), texcoord_(Vec2(0)),
      distance_(kMaxFloat), pdf_area_(kMaxFloat)
{
}

QUALIFIER_DEVICE Intersection::Intersection(const uint64_t id_instance, const float distance)
    : valid_(true), absorb_(true), inside_(false),
      id_instance_(id_instance), id_bsdf_(kInvalidId),
      pos_(Vec3(0)), normal_(Vec3(0)), texcoord_(Vec2(0)),
      distance_(distance), pdf_area_(kMaxFloat)

{
}

QUALIFIER_DEVICE Intersection::Intersection(const uint64_t id_instance,
                                            const bool inside,
                                            const Vec2 &texcoord,
                                            const Vec3 &pos,
                                            const Vec3 &normal,
                                            const float distance,
                                            const uint64_t id_bsdf,
                                            const float pdf_area)
    : valid_(true), absorb_(false), inside_(inside),
      id_instance_(id_instance), id_bsdf_(id_bsdf),
      pos_(pos), normal_(normal), texcoord_(texcoord),
      distance_(distance), pdf_area_(pdf_area)
{
}

QUALIFIER_DEVICE SamplingRecord Intersection::Sample(const Vec3 &wo, Bsdf **bsdf,
                                                     const float *pixel_buffer,
                                                     Texture **texture_buffer,
                                                     uint64_t *seed) const
{
    SamplingRecord rec;
    rec.texcoord = texcoord_;
    rec.pos = pos_;
    rec.wo = wo;
    if (bsdf != nullptr)
    {
        rec.inside = inside_;
        rec.normal = normal_;
        if (Dot(wo, normal_) < 0.0f)
        {
            rec.inside = !inside_;
            rec.normal = -normal_;
        }
        (*bsdf)->Sample(pixel_buffer, texture_buffer, seed, &rec);
    }
    else
    {
        rec.wi = wo;
        rec.pdf = 1.0f;
        rec.attenuation = Vec3(1.0f);
        rec.valid = true;
    }
    return rec;
}

QUALIFIER_DEVICE SamplingRecord Intersection::Evaluate(const Vec3 &wi, const Vec3 &wo,
                                                       Bsdf **bsdf, const float *pixel_buffer,
                                                       Texture **texture_buffer,
                                                       uint64_t *seed) const
{
    SamplingRecord rec;
    rec.texcoord = texcoord_;
    rec.pos = pos_;
    rec.wo = wo;
    rec.wi = wi;
    if (bsdf)
    {
        rec.inside = inside_;
        rec.normal = normal_;
        if (Dot(-wi, normal_) < 0.0f)
        {
            rec.inside = !inside_;
            rec.normal = -normal_;
        }
        (*bsdf)->Evaluate(pixel_buffer, texture_buffer, seed, &rec);
    }
    else
    {
        rec.pdf = 1;
        rec.attenuation = Vec3(1);
        rec.valid = true;
    }
    return rec;
}