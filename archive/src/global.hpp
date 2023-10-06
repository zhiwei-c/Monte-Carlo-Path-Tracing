#pragma once

#include <limits>
#include <vector>
#include <string>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

#define WATER_TIGHT

#if !defined(NAMESPACE_BEGIN)
#define NAMESPACE_BEGIN(name) \
    namespace name            \
    {
#endif

#if !defined(NAMESPACE_END)
#define NAMESPACE_END(name) }
#endif

NAMESPACE_BEGIN(raytracer)

using dvec2 = glm::dvec2;
using dvec3 = glm::dvec3;
using u32vec3 = glm::u32vec3;
using dvec4 = glm::dvec4;
using dmat3 = glm::dmat3;
using dmat4 = glm::dmat4;

class Accelerator;
class Bsdf;
class Emitter;
class Integrator;
class Medium;
class PhaseFunction;
class Ndf;
class Sampler;
class Shape;
class Texture;

constexpr size_t Hash(const char *str)
{
    return (*str ? Hash(str + 1) * 256 : 0) + static_cast<size_t>(*str);
}

constexpr size_t operator"" _hash(const char *str, size_t)
{
    return Hash(str);
}

NAMESPACE_END(raytracer)