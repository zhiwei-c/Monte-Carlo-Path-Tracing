#include "instance.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE Instance::Instance()
    : id_instance_(kInvalidId), index_offset_(0), num_primitive_(0)
{
}

QUALIFIER_DEVICE Instance::Instance(const uint32_t id_instance, const Instance::Info &info,
                                    Primitive *primitive_buffer)
    : id_instance_(id_instance), index_offset_(info.index_offset),
      num_primitive_(info.num_primitive)
{
    area_ = 0.0f;
    for (uint32_t i = 0; i < num_primitive_; ++i)
        area_ += primitive_buffer[index_offset_ + i].area();

    const float pdf_area = 1.0f / area_;
    for (uint32_t i = 0; i < num_primitive_; ++i)
    {
        primitive_buffer[index_offset_ + i].SetIdInstance(id_instance);
        primitive_buffer[index_offset_ + i].SetPdfArea(pdf_area);
        aabb_ += primitive_buffer[index_offset_ + i].aabb();
    }
}

QUALIFIER_DEVICE void Instance::SamplePoint(Primitive *primitive_buffer, uint32_t *seed,
                                            Intersection *its) const
{
    const float thresh = RandomFloat(seed) * area_;
    float area = 0.0f;
    for (uint32_t i = 0; i < num_primitive_; ++i)
    {
        area += primitive_buffer[index_offset_ + i].area();
        if (thresh <= area)
        {
            primitive_buffer[index_offset_ + i].SamplePoint(seed, its);
            return;
        }
    }
}