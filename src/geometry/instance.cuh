#pragma once

#include "primitive.cuh"
#include "../accelerators/aabb.cuh"
#include "../tensor/tensor.cuh"

class Instance
{
public:
    struct Info
    {
        bool is_emitter;
        uint32_t index_offset;
        uint32_t num_primitive;
    };

    QUALIFIER_DEVICE Instance();
    QUALIFIER_DEVICE Instance(const uint32_t id_instance, const Instance::Info &info,
                              Primitive *primitive_buffer);

    QUALIFIER_DEVICE void SamplePoint(Primitive *primitive_buffer, uint32_t *seed,
                                      Intersection *its) const;

private:
    uint32_t id_instance_;
    uint32_t num_primitive_;
    uint32_t index_offset_;
    float area_;
    AABB aabb_;
};