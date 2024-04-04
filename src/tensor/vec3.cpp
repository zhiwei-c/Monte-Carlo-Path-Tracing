#include "csrt/tensor/vec3.hpp"

#include <cmath>

namespace csrt
{

QUALIFIER_D_H Uvec3::Uvec3(const uint32_t x, const uint32_t y, const uint32_t z)
    : x(x), y(y), z(z)
{
}

QUALIFIER_D_H uint32_t &Uvec3::operator[](const int i)
{
    switch (i)
    {
    case 0:
        return x;
    case 1:
        return y;
    default:
        return z;
    }
}

QUALIFIER_D_H uint32_t Uvec3::operator[](const int i) const
{
    switch (i)
    {
    case 0:
        return x;
    case 1:
        return y;
    default:
        return z;
    }
}

QUALIFIER_D_H Vec3::Vec3(const float x, const float y, const float z)
    : x(x), y(y), z(z)
{
}

QUALIFIER_D_H float &Vec3::operator[](const int i)
{
    switch (i)
    {
    case 0:
        return x;
    case 1:
        return y;
    default:
        return z;
    }
}

QUALIFIER_D_H float Vec3::operator[](const int i) const
{
    switch (i)
    {
    case 0:
        return x;
    case 1:
        return y;
    default:
        return z;
    }
}

QUALIFIER_D_H void Vec3::operator=(const Vec3 &vec)
{
    x = vec.x, y = vec.y, z = vec.z;
}

QUALIFIER_D_H Vec3 &Vec3::operator+=(const Vec3 &vec)
{
    x += vec.x, y += vec.y, z += vec.z;
    return *this;
}

QUALIFIER_D_H Vec3 &Vec3::operator-=(const Vec3 &vec)
{
    x -= vec.x, y -= vec.y, z -= vec.z;
    return *this;
}

QUALIFIER_D_H Vec3 &Vec3::operator*=(const Vec3 &vec)
{
    x *= vec.x, y *= vec.y, z *= vec.z;
    return *this;
}

QUALIFIER_D_H Vec3 &Vec3::operator/=(const Vec3 &vec)
{
    const float k0 = 1.0f / vec.x, k1 = 1.0f / vec.y, k2 = 1.0f / vec.z;
    x *= k0, y *= k1, z *= k2;
    return *this;
}

QUALIFIER_D_H Vec3 &Vec3::operator*=(const float t)
{
    x *= t, y *= t, z *= t;
    return *this;
}

QUALIFIER_D_H Vec3 &Vec3::operator/=(const float t)
{
    const float k = 1.0f / t;
    x *= k, y *= k, z *= k;
    return *this;
}

QUALIFIER_D_H Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

QUALIFIER_D_H Vec3 operator-(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
}

QUALIFIER_D_H Vec3 operator*(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
}

QUALIFIER_D_H Vec3 operator/(const Vec3 &v1, const Vec3 &v2)
{
    const float k0 = 1.0f / v2.x, k1 = 1.0f / v2.y, k2 = 1.0f / v2.z;
    return {v1.x * k0, v1.y * k1, v1.z * k2};
}

QUALIFIER_D_H Vec3 operator+(const Vec3 &vec, const float t)
{
    return {vec.x + t, vec.y + t, vec.z + t};
}

QUALIFIER_D_H Vec3 operator-(const Vec3 &vec, const float t)
{
    return {vec.x - t, vec.y - t, vec.z - t};
}

QUALIFIER_D_H Vec3 operator*(const Vec3 &vec, const float t)
{
    return {vec.x * t, vec.y * t, vec.z * t};
}

QUALIFIER_D_H Vec3 operator/(const Vec3 &vec, const float t)
{
    const float k = 1.0f / t;
    return {vec.x * k, vec.y * k, vec.z * k};
}

QUALIFIER_D_H Vec3 operator+(const float t, const Vec3 &vec)
{
    return {t + vec.x, t + vec.y, t + vec.z};
}

QUALIFIER_D_H Vec3 operator-(const float t, const Vec3 &vec)
{
    return {t - vec.x, t - vec.y, t - vec.z};
}

QUALIFIER_D_H Vec3 operator*(const float t, const Vec3 &vec)
{
    return {t * vec.x, t * vec.y, t * vec.z};
}

QUALIFIER_D_H Vec3 operator/(const float t, const Vec3 &vec)
{
    const float k0 = 1.0f / vec.x, k1 = 1.0f / vec.y, k2 = 1.0f / vec.z;
    return {t * k0, t * k1, t * k2};
}

QUALIFIER_D_H float Length(const Vec3 &vec)
{
    return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

QUALIFIER_D_H Vec3 Normalize(const Vec3 &vec)
{
    const float k = 1.0f / Length(vec);
    return vec * k;
}

QUALIFIER_D_H float Dot(const Vec3 &v1, const Vec3 &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

QUALIFIER_D_H Vec3 Cross(const Vec3 &v1, const Vec3 &v2)
{
    return {v1.y * v2.z - v1.z * v2.y, -v1.x * v2.z + v1.z * v2.x,
            v1.x * v2.y - v1.y * v2.x};
}

QUALIFIER_D_H Vec3 Min(const Vec3 &v1, const Vec3 &v2)
{
    return {fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z)};
}

QUALIFIER_D_H Vec3 Max(const Vec3 &v1, const Vec3 &v2)
{
    return {fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z)};
}

QUALIFIER_D_H Vec3 Sqrt(const Vec3 &v)
{
    return {sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)};
}

} // namespace csrt