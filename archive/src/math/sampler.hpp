#pragma once

#include <random>

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

class Sampler
{
public:
    Sampler(unsigned int seed)
    {
        e_.seed(seed);
        dist_ = std::uniform_real_distribution<double>(0, 1);
    }

    double Next1D()
    {
        return dist_(e_);
    }

    dvec2 Next2D()
    {
        return dvec2{Next1D(), Next1D()};
    }

private:
    std::minstd_rand e_;
    std::uniform_real_distribution<double> dist_;
};

NAMESPACE_END(raytracer)