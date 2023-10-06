#pragma once

#include "aabb.hpp"
#include "../core/intersection.hpp"
#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

class Accelerator
{
public:
    virtual ~Accelerator() {}

    virtual bool Intersect(const Ray &ray, Sampler* sampler, Intersection *its) const = 0;
    virtual Intersection SamplePoint(Sampler* sampler) const = 0;

    virtual double area() const = 0;
    virtual AABB aabb() const = 0;

protected:
    Accelerator() {}
};

NAMESPACE_END(raytracer)