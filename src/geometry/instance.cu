#include "instance.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE Instance::Instance()
    : id_(kInvalidId), is_emitter_(false), num_primitives_(0), index_offset_(0), aabb_(AABB())
{
}

QUALIFIER_DEVICE Instance::Instance(const uint64_t id, const bool is_emitter,
                                    const uint64_t index_offset, const uint64_t num_primitives,
                                    Primitive *primitive_buffer)
    : id_(id), is_emitter_(is_emitter), index_offset_(index_offset), num_primitives_(num_primitives)
{
    area_ = 0.0f;
    for (uint64_t i = 0; i < num_primitives_; ++i)
        area_ += primitive_buffer[index_offset + i].GetArea();

    const float pdf_area = 1.0f / area_;
    for (uint64_t i = 0; i < num_primitives_; ++i)
    {
        primitive_buffer[index_offset + i].SetIdInstance(id);
        primitive_buffer[index_offset + i].SetPdfArea(pdf_area);
        aabb_ += primitive_buffer[index_offset + i].GetAabb();
    }
}

QUALIFIER_DEVICE void Instance::SamplePoint(const Primitive *primitive_buffer, uint64_t *seed,
                                            Intersection *its) const
{
    const float thresh = RandomFloat(seed) * area_;
    float area = 0.0f;
    for (uint64_t i = 0; i < num_primitives_; ++i)
    {
        area += primitive_buffer[index_offset_ + i].GetArea();
        if (thresh <= area)
        {
            primitive_buffer[index_offset_ + i].SamplePoint(seed, its);
            return;
        }
    }
}