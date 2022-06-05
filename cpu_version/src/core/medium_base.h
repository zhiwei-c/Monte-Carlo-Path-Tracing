#pragma once

#include <memory>
#include <utility>

#include "phase_function.h"

NAMESPACE_BEGIN(raytracer)

class Medium
{
public:
    virtual ~Medium() {}

    virtual bool SampleDistance(Float max_distance, Float &distance, Float &pdf, Spectrum &attenuation) const = 0;

    virtual std::pair<Spectrum, Float> EvalDistance(bool scattered, Float distance) const = 0;

    virtual void SamplePhaseFunction(SamplingRecord &rec) const = 0;

    virtual void EvalPhaseFunction(SamplingRecord &rec) const = 0;

protected:
    Medium(std::unique_ptr<PhaseFunction> phase_function)
        : phase_function_(std::move(phase_function))
    {
    }

    std::unique_ptr<PhaseFunction> phase_function_;
};

NAMESPACE_END(raytracer)