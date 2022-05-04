#pragma once

#if !defined(NAMESPACE_BEGIN)
#define NAMESPACE_BEGIN(name) \
    namespace name            \
    {
#endif

#if !defined(NAMESPACE_END)
#define NAMESPACE_END(name) }
#endif

#include <string>

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

NAMESPACE_BEGIN(simple_renderer)

using Float = glm::f64;
using Vector3 = glm::dvec3;
using Spectrum = Vector3;
using Vector2 = glm::dvec2;
using Mat4 = glm::dmat4;
using Mat3 = glm::dmat3;

constexpr auto kEpsilon = static_cast<Float>(1e-10);
constexpr auto kOneMinusEpsilon = static_cast<Float>(1 - 1e-10);
constexpr auto kEpsilonDistance = static_cast<Float>(1e-6);
constexpr auto kEpsilonL = static_cast<Float>(1e-3);
constexpr auto kEpsilonPdf = static_cast<Float>(1e-5);
constexpr auto kEpsilonPdf2 = static_cast<Float>(5e-2);

constexpr auto kEpsilonMachine = std::numeric_limits<Float>::epsilon() * 0.5;

inline Float GammaError(int n)
{
    return (n * kEpsilonMachine) / (1 - n * kEpsilonMachine);
}

constexpr size_t Hash(const char *str)
{
    return (*str ? Hash(str + 1) * 256 : 0) + static_cast<size_t>(*str);
}

constexpr size_t operator"" _hash(const char *str, size_t)
{
    return Hash(str);
}

NAMESPACE_END(simple_renderer)