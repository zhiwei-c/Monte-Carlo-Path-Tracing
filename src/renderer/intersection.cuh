#pragma once

#include "../tensor/tensor.cuh"
#include "../bsdfs/bsdfs.cuh"
#include "../textures/texture.cuh"

struct Intersection
{
    bool valid;
    bool absorb;
    bool inside;
    float distance;
    float pdf_area;
    uint32_t id_instance;
    uint32_t id_bsdf;
    Vec2 texcoord;
    Vec3 position;
    Vec3 tangent;
    Vec3 bitangent;
    Vec3 normal;

    QUALIFIER_DEVICE Intersection()
        : valid(false), absorb(false), inside(false), id_instance(kInvalidId), id_bsdf(kInvalidId),
          position(Vec3(0)), normal(Vec3(0)), texcoord(Vec2(0)), distance(kMaxFloat),
          pdf_area(0)
    {
    }

    QUALIFIER_DEVICE Intersection(const Vec3 &in_position)
        : valid(true), absorb(false), inside(false), id_instance(kInvalidId), id_bsdf(kInvalidId),
          position(in_position), normal(Vec3(0)), texcoord(Vec2(0)), distance(kMaxFloat),
          pdf_area(0)
    {
    }

    QUALIFIER_DEVICE Intersection(const uint32_t in_id_instance, const float in_distance)
        : valid(true), absorb(true), inside(false), id_instance(in_id_instance), id_bsdf(kInvalidId),
          position(Vec3(0)), normal(Vec3(0)), texcoord(Vec2(0)), distance(in_distance),
          pdf_area(0)
    {
    }

    QUALIFIER_DEVICE Intersection(const uint32_t in_id_instance, const Vec2 &in_texcoord,
                                  const Vec3 &in_position, const Vec3 &in_normal,
                                  const uint32_t in_id_bsdf, const float in_pdf_area)
        : valid(true), absorb(false), inside(false), id_instance(in_id_instance),
          id_bsdf(in_id_bsdf), position(in_position), normal(in_normal), texcoord(in_texcoord),
          distance(kMaxFloat), pdf_area(in_pdf_area)
    {
    }

    QUALIFIER_DEVICE Intersection(const uint32_t in_id_instance, const bool in_inside,
                                  const Vec2 &in_texcoord, const Vec3 &in_position,
                                  const Vec3 &in_normal, const Vec3 &in_tangent,
                                  const Vec3 &in_bitangent, const float in_distance,
                                  const uint32_t in_id_bsdf, const float in_pdf_area)
        : valid(true), absorb(false), inside(in_inside), id_instance(in_id_instance),
          id_bsdf(in_id_bsdf), position(in_position), normal(in_normal), texcoord(in_texcoord),
          distance(in_distance), pdf_area(in_pdf_area)
    {
    }
};
