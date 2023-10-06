#pragma once

#include "primitive.cuh"
#include "../accelerators/aabb.cuh"
#include "../tensor/tensor.cuh"

class Instance
{
public:
    QUALIFIER_DEVICE Instance();
    QUALIFIER_DEVICE Instance(const uint64_t id, const bool is_emitter, const uint64_t index_offset,
                              const uint64_t num_primitives, Primitive *primitive_buffer);

    QUALIFIER_DEVICE void SamplePoint(const Primitive *primitive_buffer, uint64_t *seed,
                                      Intersection *its) const;

    QUALIFIER_DEVICE bool IsEmitter() const { return is_emitter_; }

private:
    bool is_emitter_;
    uint64_t id_;
    uint64_t num_primitives_;
    uint64_t index_offset_;
    float area_;
    AABB aabb_;
};