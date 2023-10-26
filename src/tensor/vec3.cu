#include "tensor/vec3.cuh"

#include <cmath>

NAMESPACE_BEGIN(rt)

QUALIFIER_DEVICE Vec3 &Vec3::operator+=(const Vec3 &vec)
{
    x += vec.x, y += vec.y, z += vec.z;
    return *this;
}

QUALIFIER_DEVICE Vec3 &Vec3::operator-=(const Vec3 &vec)
{
    x -= vec.x, y -= vec.y, z -= vec.z;
    return *this;
}

QUALIFIER_DEVICE Vec3 &Vec3::operator*=(const Vec3 &vec)
{
    x *= vec.x, y *= vec.y, z *= vec.z;
    return *this;
}

QUALIFIER_DEVICE Vec3 &Vec3::operator/=(const Vec3 &vec)
{
    const float k0 = 1.0f / vec.x, k1 = 1.0f / vec.y, k2 = 1.0f / vec.z;
    x *= k0, y *= k1, z *= k2;
    return *this;
}

QUALIFIER_DEVICE Vec3 &Vec3::operator*=(const float t)
{
    x *= t, y *= t, z *= t;
    return *this;
}

QUALIFIER_DEVICE Vec3 &Vec3::operator/=(const float t)
{
    const float k = 1.0f / t;
    x *= k, y *= k, z *= k;
    return *this;
}

QUALIFIER_DEVICE Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

QUALIFIER_DEVICE Vec3 operator-(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
}

QUALIFIER_DEVICE Vec3 operator*(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
}

QUALIFIER_DEVICE Vec3 operator/(const Vec3 &v1, const Vec3 &v2)
{
    const float k0 = 1.0f / v2.x, k1 = 1.0f / v2.y, k2 = 1.0f / v2.z;
    return {v1.x * k0, v1.y * k1, v1.z * k2};
}

QUALIFIER_DEVICE Vec3 operator+(const Vec3 &vec, float t)
{
    return {vec.x + t, vec.y + t, vec.z + t};
}

QUALIFIER_DEVICE Vec3 operator-(const Vec3 &vec, float t)
{
    return {vec.x - t, vec.y - t, vec.z - t};
}

QUALIFIER_DEVICE Vec3 operator*(const Vec3 &vec, float t)
{
    return {vec.x * t, vec.y * t, vec.z * t};
}

QUALIFIER_DEVICE Vec3 operator/(const Vec3 &vec, float t)
{
    const float k = 1.0f / t;
    return {vec.x * k, vec.y * k, vec.z * k};
}

QUALIFIER_DEVICE Vec3 operator+(float t, const Vec3 &vec)
{
    return {t + vec.x, t + vec.y, t + vec.z};
}

QUALIFIER_DEVICE Vec3 operator-(float t, const Vec3 &vec)
{
    return {t - vec.x, t - vec.y, t - vec.z};
}

QUALIFIER_DEVICE Vec3 operator*(float t, const Vec3 &vec)
{
    return {t * vec.x, t * vec.y, t * vec.z};
}

QUALIFIER_DEVICE Vec3 operator/(float t, const Vec3 &vec)
{
    const float k0 = 1.0f / vec.x, k1 = 1.0f / vec.y, k2 = 1.0f / vec.z;
    return {t * k0, t * k1, t * k2};
}

QUALIFIER_DEVICE float Length(const Vec3 &vec)
{
    return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

QUALIFIER_DEVICE Vec3 Normalize(const Vec3 &vec)
{
    const float k = 1.0f / Length(vec);
    return vec * k;
}

QUALIFIER_DEVICE float Dot(const Vec3 &v1, const Vec3 &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

QUALIFIER_DEVICE Vec3 Cross(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.y * v2.z - v1.z * v2.y, -v1.x * v2.z + v1.z * v2.x, v1.x * v2.y - v1.y * v2.x};
}

QUALIFIER_DEVICE Vec3 Min(const Vec3 &v1, const Vec3 &v2)
{
    return {fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z)};
}

QUALIFIER_DEVICE Vec3 Max(const Vec3 &v1, const Vec3 &v2)
{
    return {fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z)};
}

NAMESPACE_END(rt)