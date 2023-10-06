#pragma once

#include "../tensor/tensor.cuh"
#include "../bsdfs/bsdfs.cuh"
#include "../textures/texture.cuh"

class Intersection
{
public:
    QUALIFIER_DEVICE Intersection();
    QUALIFIER_DEVICE explicit Intersection(const Vec3 &pos);
    QUALIFIER_DEVICE Intersection(const uint64_t id_instance, const float distance);
    QUALIFIER_DEVICE Intersection(const uint64_t id_instance, const bool inside,
                                  const Vec2 &texcoord, const Vec3 &pos, const Vec3 &normal,
                                  const float distance, const uint64_t id_bsdf, const float pdf_area);

    QUALIFIER_DEVICE SamplingRecord Sample(const Vec3 &wo, Bsdf **bsdf,
                                           const float *pixel_buffer, Texture **texture_buffer,
                                           uint64_t *seed) const;
    QUALIFIER_DEVICE SamplingRecord Evaluate(const Vec3 &wi, const Vec3 &wo, Bsdf **bsdf,
                                             const float *pixel_buffer,
                                             Texture **texture_buffer,
                                             uint64_t *seed) const;

    QUALIFIER_DEVICE bool valid() const { return valid_; }
    QUALIFIER_DEVICE bool absorb() const { return absorb_; }
    QUALIFIER_DEVICE uint64_t id_bsdf() const { return id_bsdf_; }
    QUALIFIER_DEVICE Vec3 pos() const { return pos_; }
    QUALIFIER_DEVICE Vec3 normal() const { return normal_; }
    QUALIFIER_DEVICE Vec2 texcoord() const { return texcoord_; }
    QUALIFIER_DEVICE float distance() const { return distance_; }
    QUALIFIER_DEVICE float pdf_area() const { return pdf_area_; }

private:
    bool valid_;
    bool absorb_;
    bool inside_;
    float distance_;
    float pdf_area_;
    uint64_t id_instance_;
    uint64_t id_bsdf_;
    Vec2 texcoord_;
    Vec3 pos_;
    Vec3 normal_;
};
