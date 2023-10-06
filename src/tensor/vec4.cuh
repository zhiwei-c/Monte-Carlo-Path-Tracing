#pragma once

#include "vec3.cuh"

struct Vec4
{

    float x, y, z, w;

    QUALIFIER_DEVICE Vec4() : x(0), y(0), z(0), w(0) {}

    QUALIFIER_DEVICE Vec4(float x) : x(x), y(x), z(x), w(x) {}

    QUALIFIER_DEVICE Vec4(float x, float y, float z) : x(x), y(y), z(z), w(0) {}

    QUALIFIER_DEVICE Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    QUALIFIER_DEVICE Vec4(Vec3 v) : x(v.x), y(v.y), z(v.z), w(0) {}

    QUALIFIER_DEVICE Vec4(Vec3 v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

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
        case 2:
            return z;
            break;
        default:
            return w;
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
        case 2:
            return z;
            break;
        default:
            return w;
            break;
        }
    }

    QUALIFIER_DEVICE void operator=(const Vec4 &v1)
    {
        x = v1.x;
        y = v1.y;
        z = v1.z;
        w = v1.w;
    }

    QUALIFIER_DEVICE const Vec4 &operator+() const { return *this; }
    QUALIFIER_DEVICE Vec4 operator-() const { return Vec4(-x, -y, -z, -w); }

    QUALIFIER_DEVICE Vec4 &operator+=(const Vec4 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }

    QUALIFIER_DEVICE Vec4 &operator-=(const Vec4 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }

    QUALIFIER_DEVICE Vec4 &operator*=(const Vec4 &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }

    QUALIFIER_DEVICE Vec4 &operator/=(const Vec4 &v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        w /= v.w;
        return *this;
    }

    QUALIFIER_DEVICE Vec4 &operator*=(const float t)
    {
        x *= t;
        y *= t;
        z *= t;
        w *= t;
        return *this;
    }

    QUALIFIER_DEVICE Vec4 &operator/=(const float t)
    {
        const float k = 1.0f / t;
        x *= k;
        y *= k;
        z *= k;
        w *= k;
        return *this;
    }
};

inline QUALIFIER_DEVICE Vec4 operator+(const Vec4 &v1, const Vec4 &v2)
{
    return Vec4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline QUALIFIER_DEVICE Vec4 operator-(const Vec4 &v1, const Vec4 &v2)
{
    return Vec4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

inline QUALIFIER_DEVICE Vec4 operator*(const Vec4 &v1, const Vec4 &v2)
{
    return Vec4(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w);
}

inline QUALIFIER_DEVICE Vec4 operator/(const Vec4 &v1, const Vec4 &v2)
{
    return Vec4(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w);
}

inline QUALIFIER_DEVICE Vec4 operator*(float t, const Vec4 &v)
{
    return Vec4(t * v.x, t * v.y, t * v.z, t * v.w);
}

inline QUALIFIER_DEVICE Vec4 operator/(Vec4 v, float t)
{
    const float k = 1.0f / t;
    return Vec4(v.x * k, v.y * k, v.z * k, v.w * k);
}

inline QUALIFIER_DEVICE Vec4 operator*(const Vec4 &v, float t)
{
    return Vec4(t * v.x, t * v.y, t * v.z, t * v.w);
}

inline QUALIFIER_DEVICE float Dot(const Vec4 &v1, const Vec4 &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}
