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

NAMESPACE_BEGIN(raytracer)

using Float = glm::f64;
using Vector3 = glm::dvec3;
using Spectrum = Vector3;
using Vector2 = glm::dvec2;
using Mat4 = glm::dmat4;
using Mat3 = glm::dmat3;

constexpr Float kPi = glm::pi<Float>();
constexpr Float kPiInv = 1.0 / glm::pi<Float>();
constexpr Float kFourPiInv = 0.25 * glm::pi<Float>();

constexpr Float kMaxFloat = std::numeric_limits<Float>::max();
constexpr Float kLowestFloat = std::numeric_limits<Float>::lowest();

constexpr auto kMaxVector3 = Vector3(kMaxFloat);
constexpr auto kMinVector3 = Vector3(kLowestFloat);

constexpr Float kEpsilon = 1e-10;
constexpr Float kEpsilonDistance = 1e-6;
constexpr Float kEpsilonPdf = 1e-3;
constexpr Float kEpsilonPdf2 = 0.025 * glm::pi<Float>();
constexpr Float kEpsilonMachine = std::numeric_limits<Float>::epsilon() * 0.5;

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

NAMESPACE_END(raytracer)