#include "tensor/vec2.cuh"

#include <cmath>

NAMESPACE_BEGIN(rt)

QUALIFIER_DEVICE Vec2 &Vec2::operator+=(const Vec2 &vec)
{
    u += vec.u, v += vec.v;
    return *this;
}

QUALIFIER_DEVICE Vec2 &Vec2::operator-=(const Vec2 &vec)
{
    u -= vec.u, v -= vec.v;
    return *this;
}

QUALIFIER_DEVICE Vec2 &Vec2::operator*=(const Vec2 &vec)
{
    u *= vec.u, v *= vec.v;
    return *this;
}

QUALIFIER_DEVICE Vec2 &Vec2::operator/=(const Vec2 &vec)
{
    const float k0 = 1.0f / vec.u, k1 = 1.0f / vec.v;
    u *= k0, v *= k1;
    return *this;
}

QUALIFIER_DEVICE Vec2 &Vec2::operator*=(const float t)
{
    u *= t, v *= t;
    return *this;
}

QUALIFIER_DEVICE Vec2 &Vec2::operator/=(const float t)
{
    const float k = 1.0f / t;
    u *= k, v *= k;
    return *this;
}

QUALIFIER_DEVICE float Vec2::Length()
{
    return sqrtf(u * u + v * v);
}

QUALIFIER_DEVICE Vec2 Vec2::Normalize()
{
    const float k = 1.0f / Length();
    return *this * k;
}

QUALIFIER_DEVICE Vec2 operator+(const Vec2 &v1, const Vec2 &v2)
{
    return {v1.u + v2.u, v1.v + v2.v};
}

QUALIFIER_DEVICE Vec2 operator-(const Vec2 &v1, const Vec2 &v2)
{
    return {v1.u - v2.u, v1.v - v2.v};
}

QUALIFIER_DEVICE Vec2 operator*(const Vec2 &v1, const Vec2 &v2)
{
    return {v1.u * v2.u, v1.v * v2.v};
}

QUALIFIER_DEVICE Vec2 operator/(const Vec2 &v1, const Vec2 &v2)
{
    const float k0 = 1.0f / v2.u, k1 = 1.0f / v2.v;
    return {v1.u * k0, v1.v * k1};
}

QUALIFIER_DEVICE Vec2 operator+(const Vec2 &vec, float t)
{
    return {vec.u + t, vec.v + t};
}

QUALIFIER_DEVICE Vec2 operator-(const Vec2 &vec, float t)
{
    return {vec.u - t, vec.v - t};
}

QUALIFIER_DEVICE Vec2 operator*(const Vec2 &vec, float t)
{
    return {vec.u * t, vec.v * t};
}

QUALIFIER_DEVICE Vec2 operator/(const Vec2 &vec, float t)
{
    const float k = 1.0f / t;
    return {vec.u * k, vec.v * k};
}

QUALIFIER_DEVICE Vec2 operator+(float t, const Vec2 &vec)
{
    return {t + vec.u, t + vec.v};
}

QUALIFIER_DEVICE Vec2 operator-(float t, const Vec2 &vec)
{
    return {t - vec.u, t - vec.v};
}

QUALIFIER_DEVICE Vec2 operator*(float t, const Vec2 &vec)
{
    return {t * vec.u, t * vec.v};
}

QUALIFIER_DEVICE Vec2 operator/(float t, const Vec2 &vec)
{
    const float k0 = 1.0f / vec.u, k1 = 1.0f / vec.v;
    return {t * k0, t * k1};
}

QUALIFIER_DEVICE float Dot(const Vec2 &v1, const Vec2 &v2)
{
    return v1.u * v2.u + v1.v * v2.v;
}

QUALIFIER_DEVICE Vec2 Min(const Vec2 &v1, const Vec2 &v2)
{
    return {fminf(v1.u, v2.u), fminf(v1.v, v2.v)};
}

QUALIFIER_DEVICE Vec2 Max(const Vec2 &v1, const Vec2 &v2)
{
    return {fmaxf(v1.u, v2.u), fmaxf(v1.v, v2.v)};
}

NAMESPACE_END(rt)