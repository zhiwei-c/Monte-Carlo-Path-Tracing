#pragma once

#include "../utils/math.h"
#include "sampling_record.h"

NAMESPACE_BEGIN(raytracer)

class PhaseFunction
{
public:
    virtual ~PhaseFunction() {}

    virtual void Sample(SamplingRecord &rec) const = 0;

    virtual void Eval(SamplingRecord &rec) const = 0;
};

NAMESPACE_END(raytracer)