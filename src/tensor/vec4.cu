#include "tensor/vec4.cuh"

#include <cmath>

NAMESPACE_BEGIN(rt)

QUALIFIER_DEVICE Vec4 &Vec4::operator+=(const Vec4 &vec)
{
    x += vec.x, y += vec.y, z += vec.z, w += vec.w;
    return *this;
}

QUALIFIER_DEVICE Vec4 &Vec4::operator-=(const Vec4 &vec)
{
    x -= vec.x, y -= vec.y, z -= vec.z, w -= vec.w;
    return *this;
}

QUALIFIER_DEVICE Vec4 &Vec4::operator*=(const Vec4 &vec)
{
    x *= vec.x, y *= vec.y, z *= vec.z, w *= vec.w;
    return *this;
}

QUALIFIER_DEVICE Vec4 &Vec4::operator/=(const Vec4 &vec)
{
    const float k0 = 1.0f / vec.x, k1 = 1.0f / vec.y, k2 = 1.0f / vec.z, k3 = 1.0f / vec.w;
    x *= k0, y *= k1, z *= k2, w *= k3;
    return *this;
}

QUALIFIER_DEVICE Vec4 &Vec4::operator*=(const float t)
{
    x *= t, y *= t, z *= t, w *= t;
    return *this;
}

QUALIFIER_DEVICE Vec4 &Vec4::operator/=(const float t)
{
    const float k = 1.0f / t;
    x *= k, y *= k, z *= k, w *= k;
    return *this;
}

QUALIFIER_DEVICE Vec4 operator+(const Vec4 &v1, const Vec4 &v2)
{
    return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w};
}

QUALIFIER_DEVICE Vec4 operator-(const Vec4 &v1, const Vec4 &v2)
{
    return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w};
}

QUALIFIER_DEVICE Vec4 operator*(const Vec4 &v1, const Vec4 &v2)
{
    return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w};
}

QUALIFIER_DEVICE Vec4 operator/(const Vec4 &v1, const Vec4 &v2)
{
    const float k0 = 1.0f / v2.x, k1 = 1.0f / v2.y, k2 = 1.0f / v2.z, k3 = 1.0f / v2.w;
    return {v1.x * k0, v1.y * k1, v1.z * k2, v1.w * k3};
}

QUALIFIER_DEVICE Vec4 operator+(const Vec4 &vec, float t)
{
    return {vec.x + t, vec.y + t, vec.z + t, vec.w + t};
}

QUALIFIER_DEVICE Vec4 operator-(const Vec4 &vec, float t)
{
    return {vec.x - t, vec.y - t, vec.z - t, vec.w - t};
}

QUALIFIER_DEVICE Vec4 operator*(const Vec4 &vec, float t)
{
    return {vec.x * t, vec.y * t, vec.z * t, vec.w * t};
}

QUALIFIER_DEVICE Vec4 operator/(const Vec4 &vec, float t)
{
    const float k = 1.0f / t;
    return {vec.x * k, vec.y * k, vec.z * k, vec.w * k};
}

QUALIFIER_DEVICE Vec4 operator+(float t, const Vec4 &vec)
{
    return {t + vec.x, t + vec.y, t + vec.z, t + vec.w};
}

QUALIFIER_DEVICE Vec4 operator-(float t, const Vec4 &vec)
{
    return {t - vec.x, t - vec.y, t - vec.z, t - vec.w};
}

QUALIFIER_DEVICE Vec4 operator*(float t, const Vec4 &vec)
{
    return {t * vec.x, t * vec.y, t * vec.z, t * vec.w};
}

QUALIFIER_DEVICE Vec4 operator/(float t, const Vec4 &vec)
{
    const float k0 = 1.0f / vec.x, k1 = 1.0f / vec.y, k2 = 1.0f / vec.z, k3 = 1.0f / vec.w;
    return {t * k0, t * k1, t * k2, t * k3};
}

QUALIFIER_DEVICE float Dot(const Vec4 &v1, const Vec4 &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

QUALIFIER_DEVICE Vec4 Min(const Vec4 &v1, const Vec4 &v2)
{
    return {fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z), fminf(v1.w, v2.w)};
}

QUALIFIER_DEVICE Vec4 Max(const Vec4 &v1, const Vec4 &v2)
{
    return {fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z), fmaxf(v1.w, v2.w)};
}

NAMESPACE_END(rt)