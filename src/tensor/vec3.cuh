#pragma once

#include "vec2.cuh"

#include <cmath>

struct Uvec3
{
    unsigned int x, y, z;

    QUALIFIER_DEVICE Uvec3() : x(0), y(0), z(0) {}

    QUALIFIER_DEVICE Uvec3(unsigned int x) : x(x), y(x), z(x) {}

    QUALIFIER_DEVICE Uvec3(unsigned int x, unsigned int y, unsigned int z) : x(x), y(y), z(z) {}

    QUALIFIER_DEVICE unsigned int &operator[](int i)
    {
        switch (i)
        {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        default:
            return z;
            break;
        }
    }

    QUALIFIER_DEVICE unsigned int operator[](int i) const
    {
        switch (i)
        {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        default:
            return z;
            break;
        }
    }
};

struct Vec3
{

    float x, y, z;

    QUALIFIER_DEVICE Vec3() : x(0), y(0), z(0) {}

    QUALIFIER_DEVICE Vec3(float x) : x(x), y(x), z(x) {}

    QUALIFIER_DEVICE Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    QUALIFIER_DEVICE float &operator[](int i)
    {
        switch (i)
        {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        default:
            return z;
            break;
        }
    }

    QUALIFIER_DEVICE float operator[](int i) const
    {
        switch (i)
        {
        case 0:
            return x;
            break;
        case 1:
            return y;
            break;
        default:
            return z;
            break;
        }
    }

    QUALIFIER_DEVICE void operator=(const Vec3 &v1)
    {
        x = v1.x;
        y = v1.y;
        z = v1.z;
    }

    QUALIFIER_DEVICE const Vec3 &operator+() const { return *this; }
    QUALIFIER_DEVICE Vec3 operator-() const { return Vec3(-x, -y, -z); }

    QUALIFIER_DEVICE Vec3 &operator+=(const Vec3 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    QUALIFIER_DEVICE Vec3 &operator-=(const Vec3 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    QUALIFIER_DEVICE Vec3 &operator*=(const Vec3 &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    QUALIFIER_DEVICE Vec3 &operator/=(const Vec3 &v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }

    QUALIFIER_DEVICE Vec3 &operator*=(const float t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    QUALIFIER_DEVICE Vec3 &operator/=(const float t)
    {
        const float k = 1.0f / t;
        x *= k;
        y *= k;
        z *= k;
        return *this;
    }

    QUALIFIER_DEVICE float Length() const
    {
        return fmaxf(sqrt(x * x + y * y + z * z), kEpsilonFloat);
    }

    QUALIFIER_DEVICE float SquaredLength() const
    {
        return fmaxf(x * x + y * y + z * z, kEpsilonFloat);
    }
};

inline QUALIFIER_DEVICE Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline QUALIFIER_DEVICE Vec3 operator-(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline QUALIFIER_DEVICE Vec3 operator*(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline QUALIFIER_DEVICE Vec3 operator/(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

inline QUALIFIER_DEVICE Vec3 operator+(const Vec3 &v, float t)
{
    return Vec3(t + v.x, t + v.y, t + v.z);
}

inline QUALIFIER_DEVICE Vec3 operator+(float t, const Vec3 &v)
{
    return Vec3(t + v.x, t + v.y, t + v.z);
}

inline QUALIFIER_DEVICE Vec3 operator-(float t, const Vec3 &v)
{
    return Vec3(t - v.x, t - v.y, t - v.z);
}

inline QUALIFIER_DEVICE Vec3 operator-(const Vec3 &v, float t)
{
    return Vec3(v.x - t, v.y - t, v.z - t);
}

inline QUALIFIER_DEVICE Vec3 operator*(float t, const Vec3 &v)
{
    return Vec3(t * v.x, t * v.y, t * v.z);
}

inline QUALIFIER_DEVICE Vec3 operator/(const Vec3 &v, float t)
{
    const float k = 1.0f / t;
    return Vec3(v.x * k, v.y * k, v.z * k);
}

inline QUALIFIER_DEVICE Vec3 operator*(const Vec3 &v, float t)
{
    return Vec3(t * v.x, t * v.y, t * v.z);
}

inline QUALIFIER_DEVICE float Dot(const Vec3 &v1, const Vec3 &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline QUALIFIER_DEVICE Vec3 Cross(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3((v1.y * v2.z - v1.z * v2.y), (-(v1.x * v2.z - v1.z * v2.x)), (v1.x * v2.y - v1.y * v2.x));
}

inline QUALIFIER_DEVICE Vec3 Min(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z));
}

inline QUALIFIER_DEVICE Vec3 Max(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z));
}

inline QUALIFIER_DEVICE Vec3 Normalize(const Vec3 &v)
{
    return v / v.Length();
}

inline QUALIFIER_DEVICE float Length(const Vec3 &v)
{
    return v.Length();
}

inline QUALIFIER_DEVICE Vec3 Sqr(const Vec3 &v)
{
    return v * v;
}

inline QUALIFIER_DEVICE Vec3 Sqrt(const Vec3 &v)
{
    return {sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)};
}
